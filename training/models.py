#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024-2025 Apple Inc. All Rights Reserved.
#
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


import copy
import os
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc
from training.utils import get_epipolar_dist, get_warped_features
import torch.nn.functional as F

def get_epipolar_attn(epipolar_corr, epipolar_mixing, patch_size=1):
    epipolar_corr = epipolar_corr.unsqueeze(1)
    mixing = epipolar_mixing[0].reshape((1, -1, 1, 1))
    temperature = epipolar_mixing[1].reshape((1, -1, 1, 1)).exp()
    cutoff = patch_size / 2 ** 0.5 + epipolar_mixing[2].reshape((1, -1, 1, 1))
    bias = epipolar_mixing[3].reshape((1, -1, 1, 1)) if epipolar_mixing.shape[0] > 3 else torch.zeros_like(mixing)
    epipolar_w = mixing * torch.sigmoid(temperature * (cutoff - epipolar_corr)) + bias
    return epipolar_w

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / (((1 - t) ** 2 + t ** 2) ** 0.5)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            B, C, H, W = x.shape
            S = H * W
            D_head = C // self.num_heads
            
            qkv = self.attn_qkv(x)
            qkv = qkv.view(B, self.num_heads, D_head * 3, S).permute(0, 1, 3, 2) # Shape: [B, n_heads, S, D_head*3]
            q, k, v = qkv.chunk(3, dim=-1) # Shape: [B, n_heads, S, D_head] each
            
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            
            y = y.permute(0, 1, 3, 2).reshape(B, C, H, W)
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x
    
#----------------------------------------------------------------------------
# U-Net encoder/decoder cross-attention block with optional self-attention (Figure 21).

@persistence.persistent_class
class XAttnBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
        epipolar_attention_bias          = False,
        imsize              = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.imsize = imsize
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.x_attn_kv = MPConv(out_channels, out_channels * 2, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None
        self.epipolar_mixing = torch.nn.Parameter(torch.zeros((4, self.num_heads, ))) if self.num_heads != 0 and epipolar_attention_bias else None


    def forward(self, x, features, emb, geometry=None):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            B, C, H, W = x.shape
            S_self = H * W
            D_head = C // self.num_heads

            # Get Q, K, V for self-attention from the decoder stream (x)
            qkv_self = self.attn_qkv(x)
            qkv_self = qkv_self.view(B, self.num_heads, D_head * 3, S_self).permute(0, 1, 3, 2)  # .view is we flatten the spatial dims H and W into sequence length S . we also separate channels into num_heads and head_dim
            # we swap the last two dimensions, namely Shape: [B, 2, 256, 192] -> [B, n_heads, S, D_head*3]
            q, k_self, v_self = qkv_self.chunk(3, dim=-1)

            # Get K, V for cross-attention from the encoder stream (features)
            S_cross = features.shape[2] * features.shape[3]
            kv_cross = self.x_attn_kv(features)
            kv_cross = kv_cross.view(B, self.num_heads, D_head * 2, S_cross).permute(0, 1, 3, 2)
            k_cross, v_cross = kv_cross.chunk(2, dim=-1)

            # Combine keys and values for joint attention
            k = torch.cat([k_self, k_cross], dim=2)
            v = torch.cat([v_self, v_cross], dim=2)

            # Use the memory-efficient scaled dot-product attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            
            
            y = y.permute(0, 1, 3, 2).reshape(B, C, H, W)
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        extra_attn          = None,         # Force attention at the start and end of every level
        epipolar_attention_bias          = False,
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.cnoise = cnoise
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                attn = (res in attn_resolutions or (extra_attn is not None and extra_attn == idx and level != 0))
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=attn, **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                attn = (res in attn_resolutions or (extra_attn is not None and extra_attn == num_blocks - idx and level != 0))
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=attn, **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

    def forward(self, x, noise_labels, geometry):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(geometry), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x
    
#----------------------------------------------------------------------------
# EDM2 X-Attention U-Net model

@persistence.persistent_class
class XAttnUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                         # Image resolution.
        img_channels,                           # Image channels.
        label_dim,                              # Class label dimensionality. 0 = unconditional.
        model_channels          = 192,          # Base multiplier for the number of channels.
        channel_mult            = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise      = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb        = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks              = 3,            # Number of residual blocks per resolution.
        attn_resolutions        = [16,8],       # List of resolutions with self-attention.
        label_balance           = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance          = 0.5,          # Balance between skip connections (0) and main path (1).
        extra_attn              = None,         # Force attention at the start and end of every level
        epipolar_attention_bias = False,        # Use epipolar attention bias
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.cnoise = cnoise
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                if res in attn_resolutions or (extra_attn is not None and extra_attn == idx and level != 0):
                    self.enc[f'{res}x{res}_block{idx}'] = XAttnBlock(cin, cout, cemb, flavor='enc', attention=True, epipolar_attention_bias=epipolar_attention_bias, imsize=img_resolution, **block_kwargs)
                else:
                    self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=False, **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = XAttnBlock(cout, cout, cemb, flavor='dec', attention=True, epipolar_attention_bias=epipolar_attention_bias, imsize=img_resolution, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                if res in attn_resolutions or (extra_attn is not None and extra_attn == num_blocks - idx and level != 0):
                    self.dec[f'{res}x{res}_block{idx}'] = XAttnBlock(cin, cout, cemb, flavor='dec', attention=True, epipolar_attention_bias=epipolar_attention_bias, imsize=img_resolution, **block_kwargs)
                else:
                    self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=False, **block_kwargs)

        self.out_conv = MPConv(cout, 3, kernel=[3,3])

    def forward(self, x, features, noise_labels, geometry):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(geometry), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            if isinstance(block, XAttnBlock):
                feature = features.pop(0)
                x = block(x, feature, emb, geometry=geometry)
            else:
                x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            if isinstance(block, XAttnBlock):
                feature = features.pop(0)
                x = block(x, feature, emb, geometry=geometry)
            else:
                x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#--------------------------------------------
# Encoder Model

@persistence.persistent_class
class UNetEncoder(UNet):
    def __init__(self, *args, no_cam=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_conv = None
        self.out_gain = None
        self.no_cam = no_cam
        for name, block in reversed(self.dec.items()): # remove all unnecessary layers from original UNet 
            if block.num_heads == 0:
                self.dec[name] = None
            else:
                break

    def forward(self, x, noise_labels, geometry):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(geometry), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        features = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            if 'conv' not in name and block.num_heads > 0:
                features.append(x)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if block is None:
                break
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
            if block.num_heads > 0:
                features.append(x)
        return features

#--------------------------------------------
# Super Resolution X-Attention U-Net model

@persistence.persistent_class
class SRXAttnUNet(XAttnUNet):
    def __init__(self, img_resolution, *args, **kwargs):
        super().__init__(*args, img_resolution=img_resolution, channels_per_head=32, **kwargs)
        cout, cin = self.enc[f'{img_resolution}x{img_resolution}_conv'].weight.shape[:2]
        # Add additional low-res input to the SR model
        cin = 2 * (cin - 1) + 1
        self.enc[f'{img_resolution}x{img_resolution}_conv'] = MPConv(cin, cout, kernel=[3,3])  



#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class NVPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Image channels.
        label_dim,                      # Class label dimensionality. 0 = unconditional.
        use_fp16            = True,     # Run the model at FP16 precision?
        sigma_data          = 0.5,      # Expected standard deviation of the training data.
        logvar_channels     = 128,      # Intermediate dimensionality for uncertainty estimation.
        super_res           = False,    # Make this a SR model
        no_time_enc         = None,     # Do not use time conditioning in Encoder model
        depth_input         = False,    # Expect depthmap as an additional input channel
        warp_depth_coor     = False,    # Warp depth coordinates as additional input channels
        uncond              = None,     # Train an unconditional diffusion model
        noisy_sr            = 0.25,     # Amount of noise to add to an SR model's low-res input
        **unet_kwargs,                  # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.super_res = super_res
        self.no_time_enc = no_time_enc
        self.depth_input = depth_input
        self.warp_depth_coor = warp_depth_coor
        self.uncond = uncond
        self.noisy_sr = noisy_sr
        self.encoder = UNetEncoder(img_resolution=img_resolution, img_channels=(img_channels + int(depth_input) + logvar_channels * int(warp_depth_coor)), label_dim=label_dim, **unet_kwargs) if not self.uncond else None
        unet_class = SRXAttnUNet if super_res else XAttnUNet
        self.unet = unet_class(img_resolution=img_resolution, img_channels=(img_channels + logvar_channels * int(warp_depth_coor)), label_dim=label_dim, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        
    def forward(self, src, dst, sigma, geometry=None, conditioning_image=None, force_fp32=False, return_logvar=False, return_features=False, inject_features=None, **unet_kwargs):
        x = dst.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        geometry = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if geometry is None else geometry.to(torch.float32).reshape(-1, self.label_dim)
        geometry = geometry * int(not self.uncond)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        x_in = (c_in * x).to(dtype)
        # Add warped depth coordinates to network inputs
        if self.warp_depth_coor:
            assert src.shape[1] == 4
            depth = src[:, 3:]
            if torch.all(src[:, :3] == 0):
                src_grid, dst_grid = [torch.zeros(src.shape[:1] + (128,) + src.shape[-2:], device=src.device, dtype=src.dtype)] * 2
            else:
                src_grid, dst_grid = get_warped_features(depth, geometry, self.logvar_fourier)
            src = torch.cat([src[:, :3], src_grid], dim=1)
            x_in = torch.cat([x_in, dst_grid], dim=1)

        # Add a low-res conditioning images to an SR model
        if self.super_res:
            assert conditioning_image is not None
            conditioning_image = conditioning_image + self.noisy_sr * torch.randn_like(conditioning_image)
            x_in = torch.cat([x_in, conditioning_image], dim=1)
        
        # Features
        if not self.training and inject_features is not None: # Use precomputed features
            features = copy.deepcopy(inject_features)
        elif self.encoder is None: # Insert zeros for an unconditional models
            features = []
            for name, block in self.unet.enc.items():
                if isinstance(block, XAttnBlock):
                    res = int(name.split("x")[0])
                    features.append(torch.zeros((x_in.shape[0], block.out_channels, res, res), dtype=x_in.dtype, device=x_in.device))
            for name, block in self.unet.dec.items():
                if isinstance(block, XAttnBlock):
                    res = int(name.split("x")[0])
                    features.append(torch.zeros((x_in.shape[0], block.out_channels, res, res), dtype=x_in.dtype, device=x_in.device))
        else: # Default encoder
            features = self.encoder(src.to(dtype), c_noise * int(not self.no_time_enc), geometry, **unet_kwargs)
        if return_features:
            return features

        F_x = self.unet(x_in, features, c_noise, geometry, **unet_kwargs)
        D_x = c_skip * dst + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x

#----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
