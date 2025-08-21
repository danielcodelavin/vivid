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

import einops
import kornia
import os
import shlex
import subprocess
import pickle

import numpy as np
import torch
from torch_utils import distributed as dist
import dnnlib

def open_tensorboard_process(summary_dir: str):
        tbp = os.environ.get("TENSORBOARD_PORT")
        command = "tensorboard --logdir {} --port {} --bind_all".format(
            summary_dir, tbp
        )
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )


# Precomputed stats for geometry normalization
MEAN = torch.tensor([ 9.6681e-01, -1.6038e-04, -3.7034e-05, -1.6904e-03, -8.7718e-05,
         9.9869e-01,  3.1288e-03, -1.0794e-03,  1.0653e-05,  3.0997e-03,
         9.6691e-01,  1.2561e-02,  5.7708e+01,  5.7704e+01,  3.2000e+01,
         3.2000e+01,  5.7708e+01,  5.7704e+01,  3.2000e+01,  3.2000e+01])
STD = torch.tensor([0.1104, 0.0346, 0.2279, 0.4930, 0.0347, 0.0091, 0.0367, 0.2208, 0.2279,
        0.0368, 0.1088, 1.0751, 6.6464, 6.6511, 0.0000, 0.0000, 6.6464, 6.6511,
        0.0000, 0.0000])


def compose_K(K):
    """
    Stack intrinsic matrices into 4 element vectors
    """
    return torch.stack((K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]), -1)


def decompose_K(t):
    """
    Unstack intrinsic matrices into 3x3 matrices
    """
    K = torch.zeros(size=t.shape[:-1] + (3, 3), dtype=t.dtype, device=t.device)
    K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2] = t.unbind(-1)
    K[..., 2, 2] = 1
    return K


def compose_geometry(tgt2src, src_K, tgt_K, imsize=64):
    """
    Stack camera extrinsics and intrinsics into a flattened vector using precomputed statistics.
    Uses image size to fit the used intrinsics
    """
    mean, std = MEAN.clone().to(device=tgt2src.device, dtype=tgt2src.dtype), STD.clone().to(device=tgt2src.device, dtype=tgt2src.dtype)
    mean[12:] *= (imsize/64)
    std[12:] *= (imsize/64) ** 2
    geometry = torch.cat((tgt2src.reshape(*tgt2src.shape[:-2], 12), compose_K(src_K), compose_K(tgt_K)), -1)
    return torch.where(std > 0, (geometry - mean) / std, torch.zeros_like(geometry))


def decompose_geometry(t, imsize=64):
    """
    Unstack camera extrinsics and intrinsics from a flattened vector representation using precomputed statistics.
    Uses image size to fit the used intrinsics
    """
    mean, std = MEAN.clone().to(device=t.device, dtype=t.dtype), STD.clone().to(device=t.device, dtype=t.dtype)
    mean[12:] *= (imsize/64)
    std[12:] *= (imsize/64) ** 2
    t = (t * std) + mean
    tgt2src, src_K, tgt_K = t[..., :12].reshape(*t.shape[:-1], 3, 4), decompose_K(t[..., 12:16]), decompose_K(t[..., 16:])
    return tgt2src, src_K, tgt_K


def resize_geometry(geometry, _from, _to):
    """
    Change geometry statistics based on a changed image size
    """
    tgt2src, src_K, tgt_K = decompose_geometry(geometry, _from)
    src_K[..., :2, :] = src_K[..., :2, :] * _to / _from
    tgt_K[..., :2, :] = tgt_K[..., :2, :] * _to / _from
    return compose_geometry(tgt2src, src_K, tgt_K, _to)


def depth_prepare(x):
    """
    Transform an image to be used by a DepthAnythingV2 model
    """
    x = x / 255
    x = kornia.geometry.transform.resize(x, 518, interpolation='bicubic', align_corners=True)
    x = ((x - torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).reshape((1, -1, 1, 1))) / 
            torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).reshape((1, -1, 1, 1)))
    return x.to(torch.float16)


def get_depth(depth_model, image, shape=None):
    """
    Applies the depth model on the given image and resizes the result to the desired shape
    """
    shape = image.shape if shape is None else shape
    with torch.no_grad():
        depth = depth_model(depth_prepare(image)).to(torch.float32)[:, None]
        depth = torch.nn.functional.interpolate(depth, shape, mode="bilinear", align_corners=True)
        return depth


def add_depth(depth_model, image, src, inv_norm):
    """
    Appends the a predicted depthmap to the given image
    """
    with torch.no_grad():
        depth = get_depth(depth_model, image, src.shape[-2:])
        if inv_norm:
            depth = 1 / depth
            depth = depth / depth.amax((1, 2, 3), keepdim=True)
            depth = (depth - 0.4947) / 0.2294
        return torch.cat([src, depth], dim=1)


def expand_extrinsics(extrinsics):
    """
    Converts a 3x4 camera matrix to a 4x4
    """
    bottom = torch.tensor([0, 0, 0, 1], device=extrinsics.device, dtype=extrinsics.dtype)
    bottom = bottom.reshape((1,) * (extrinsics.ndim - 1) + (4,)).repeat(extrinsics.shape[:-2] + (1,1))
    return torch.cat([extrinsics, bottom], -2)


def get_epipolar_dist(geometry, imsize, patch_size, device):
    """
    Computes epipolar distance between two images, for pixels of size 'patch_size'

    Uses the following formula:
    Given a line $x=a + t n$, the distance to a point $p$ is:
    $|| (a - p) - ((a - p) \\cdot n) n ||$

    """
    tgt2src, src_K, tgt_K = decompose_geometry(geometry.unsqueeze(1), imsize=imsize)
    batch = tgt2src.shape[0]
    # For single image augmentations, create a minimal synthetic translation s.t. epipolar lines exist
    tgt2src[..., :2, 3] = torch.where(tgt2src[..., :2, 3] != 0 * tgt2src[..., :2, 3], tgt2src[..., :2, 3], 1e-5 * torch.randn_like(tgt2src[..., :2, 3]))
    tgt2src[..., 2, 3] = torch.where(tgt2src[..., 2, 3].abs() > 1e-5, tgt2src[..., 2, 3], 1e-1 * tgt2src[..., :2, 3].pow(2).sum(-1).sqrt() * (2 * torch.randint_like(tgt2src[..., 2, 3], 2) - 1))
    
    # Create grid
    vv, uu = torch.meshgrid(
        torch.arange(0, imsize, patch_size, device=device), torch.arange(0, imsize, patch_size, device=device), indexing="ij"
    )
    uu = uu.float() + 0.5 * patch_size  # shift to pixel center
    vv = vv.float() + 0.5 * patch_size
    uu = einops.rearrange(uu[None, None], "b c h w -> b h w c")
    vv = einops.rearrange(vv[None, None], "b c h w -> b h w c")
    dd = torch.ones_like(uu)
    grid_uvd = torch.cat((uu, vv, dd), dim=-1)
    xyz = grid_uvd.repeat([batch, 1, 1, 1]).to(tgt2src.device)
    
    xyz1 = torch.cat((xyz @ torch.linalg.inv(tgt_K).transpose(-1, -2), torch.ones_like(xyz[..., :1])), dim=-1)
    tgt_xyz = (xyz1 @ tgt2src.transpose(-1, -2))[..., :3] @ src_K.transpose(-1, -2)
    tgt_xyz = tgt_xyz / tgt_xyz[..., 2:3]
    tgt_o = tgt2src[..., :3, 3][..., None, :] @ src_K.transpose(-1, -2)
    tgt_o = tgt_o / tgt_o[..., 2:3]
    a, b = (xyz - tgt_o).reshape((batch, -1, 1, 3))[..., :2], (tgt_xyz - tgt_o).reshape((batch, 1, -1, 3))[..., :2]
    b = b / b.pow(2).sum(-1, keepdim=True).pow(0.5)
    d = (a - (a * b).sum(-1, keepdim=True) * b).pow(2).sum(-1).pow(0.5)
    return d.permute(0, 2, 1)


def warp_image(depth, geometry, image):
    """
    Warps image based on the geometry and the depth
    """
    tgt2src, src_K, tgt_K = decompose_geometry(geometry[:, None], imsize=image.shape[-2])
    points_3d = torch.cat([image, torch.ones_like(image[..., :1])], -1)
    w = points_3d @ torch.inverse(src_K).transpose(-1, -2)
    w = torch.cat([w * depth, torch.ones_like(depth)], dim=-1)
    w = w @ torch.inverse(expand_extrinsics(tgt2src)).transpose(-1, -2)
    w = w[..., :3] @ tgt_K.transpose(-1, -2)  
    warped_grid = (w / w[..., 2:])[..., :2]
    warped_grid = torch.where(torch.isnan(warped_grid), torch.zeros_like(warped_grid), warped_grid)
    return warped_grid


def get_warped_features(depth, geometry, embedder):
    with torch.no_grad():
        batch_size, _, _, imsize = depth.shape
        grid_2d = einops.rearrange(torch.stack(torch.meshgrid(
                torch.arange(0, imsize, device=depth.device, dtype=depth.dtype), 
                torch.arange(0, imsize, device=depth.device, dtype=depth.dtype), 
                indexing="ij"
            ), dim=0)[None], "b c h w -> b h w c").repeat([batch_size, 1, 1, 1]) + 0.5
        warped_grid = warp_image(einops.rearrange(depth, "b c h w -> b h w c"), geometry, grid_2d)
        # Encode grid and Warped grid
        features = einops.rearrange(embedder(grid_2d.reshape(-1))[..., :64].reshape(batch_size, imsize, imsize, 128), "b h w c -> b c h w")
        warped_features = einops.rearrange(embedder(warped_grid.reshape(-1))[..., :64].reshape(batch_size, imsize, imsize, 128), "b h w c -> b c h w")
        return features, warped_features


def resolve_model(model, name="network", key="ema", device=torch.device("cuda")):
    """
    Loads model from persistent .pkl file
    """
    if model is not None and isinstance(model, str):
        dist.print0(f'Loading {name} from {model} ...')
        with dnnlib.util.open_url(model, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        model = data[key if key in data else 'net'].eval().to(device)
        del data
    return model


def resolve_depth_model(depth_model, device=torch.device("cuda")):
    if depth_model is not None and isinstance(depth_model, str):
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }[depth_model]
        from depth_anything_v2.dpt_metric import DepthAnythingV2
        depth_model = DepthAnythingV2(**model_configs)
        depth_model.load_state_dict(torch.load(f'depth_anything_v2_metric_hypersim_{model_configs["encoder"]}.pth', map_location=device))
        depth_model.to(torch.float16).to(device).eval()
    return depth_model
