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

"""Generate random images using the given model."""

import os
import re
import time
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import torchvision
import torchvision.transforms.functional
import dnnlib
from torch_utils import distributed as dist, misc
from training.utils import add_depth, resolve_model, resolve_depth_model
from training.custom_litdata_loader import CustomLitDataset, VanillaCollate, DualSourceCollate, VANILLA_MODE

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://ml-site.cdn-apple.com/models/vivid' # TODO: fill with URLS after uploading

config_presets = {
    'vivid': dnnlib.EasyDict(net=f'{model_root}/vivid-base.pkl', sr_model=f'{model_root}/vivid-sr.pkl', gnet=f'{model_root}/vivid-uncond.pkl', guidance=1.5, range_selection="mid"),
}


def edm_sampler(
    net, src, noise, labels=None, gnet=None, conditioning_image=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def wrap_denoise():
        features = None
        if hasattr(net, "no_time_enc") and net.no_time_enc:
            features = net(src, torch.zeros_like(src), torch.ones((src.shape[0]), dtype=dtype, device=noise.device), labels, conditioning_image, return_features=True)

        def denoise(x, t):
            t = t.expand(x.shape[0])
            Dx = net(src, x, t, labels, conditioning_image, inject_features=features).to(dtype)
            if guidance == 1:
                return Dx
            # Note: Guidance with gnet might require similar handling if gnet is also dual-source
            ref_Dx = gnet(src, x, t).to(dtype)
            return ref_Dx.lerp(Dx, guidance)
        
        return denoise
    denoise = wrap_denoise()

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        denoised_output = denoise(x_hat, t_hat)
        
        
        is_dual_source = denoised_output.shape[0] != x_hat.shape[0]
        if is_dual_source:
           
            d_cur = (x_hat[::2] - denoised_output) / t_hat
            x_next_half = x_hat[::2] + (t_next - t_hat) * d_cur
           
            x_next = x_hat.clone()
            x_next[::2] = x_next_half
            x_next[1::2] = x_next_half
        else: # Original vanilla logic
            d_cur = (x_hat - denoised_output) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

       
        if i < num_steps - 1:
            denoised_prime = denoise(x_next, t_next)
            if is_dual_source:
                # Apply correction only to the first item of each pair
                d_prime = (x_next[::2] - denoised_prime) / t_next
                x_next_half = x_hat[::2] + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                x_next[::2] = x_next_half
                x_next[1::2] = x_next_half
            else: # Original vanilla logic
                d_prime = (x_next - denoised_prime) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    if is_dual_source:
        return x_next[::2]
    return x_next

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#----------------------------------------------------------------------------

def generate_images_nvs(
    net,                                            # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                     # Reference network for guidance. None = same as main network.
    encoder             = None,                     # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                     # Where to save the output images. None = do not save.
    subdirs             = False,                    # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),            # List of random seeds.
    class_idx           = None,                     # Class label. None = select randomly.
    max_batch_size      = 32,                       # Maximum batch size for the diffusion model.
    encoder_batch_size  = None,                     # Maximum batch size for the encoder. None = default.
    verbose             = True,                     # Enable status prints?
    device              = torch.device('cuda'),     # Which compute device to use.
    sampler_fn          = edm_sampler,              # Which sampler function to use.
    datakwargs          = dict(class_name='datautils.RealEstate10K'),
                                                    # Dataset class
    range_selection     = None,                     # Frame range selection override
    sr_model            = None,                     # SR model to use, can be torch model or pickle path
    depth_model         = None,                     # Depth model to use, can be torch model or size of DepthAnythingV2
    **sampler_kwargs,                               # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data[('ema' if 'ema' in data else 'net')].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
        del data
    assert net is not None


    # Load guidance network.
    gnet = resolve_model(gnet, name="guidance")
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # initialize SR model
    sr_model = resolve_model(sr_model, name="SR")

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    
    if dist.get_rank() == 0:
        dist.print0('Setting up test dataloader for image generation...')

    job_id = os.environ.get("SLURM_JOB_ID", f"generate_{int(time.time())}")
    cache_dir = f"/dev/shm/litdata_cache_{job_id}_generate"
    if dist.get_rank() == 0:
        os.makedirs(cache_dir, exist_ok=True)
    dist.barrier()
    
    dataset_obj = CustomLitDataset(path=datakwargs['path'], cache_dir=cache_dir)
    
    if VANILLA_MODE:
        collate_fn = VanillaCollate()
    else:
        collate_fn = DualSourceCollate()
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        batch_size=max_batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    dataset_iterator = iter(dataloader)

    # Is current model an SR model?
    super_res = (net.img_resolution == 256)
    if sr_model is not None:
        # No CFG in SR Model
        sr_sampler_kwargs = {k:v for k, v in sampler_kwargs.items() if k != "guidance"}

    depth_model = resolve_depth_model(depth_model, device=device)

    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')


    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, src=None, tgt=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:
                    try:
                        data = next(dataset_iterator)
                        if data is None: continue
                    except StopIteration:
                        continue
                    
                    if VANILLA_MODE:
                        r.src, r.tgt, geometry = (data[k] for k in ["src_image", "tgt_image", "geometry"])
                        
                        num_to_process = min(len(r.seeds), r.src.shape[0])
                        if num_to_process == 0: continue
                        r.seeds = r.seeds[:num_to_process]
                        r.src = r.src[:num_to_process]
                        r.tgt = r.tgt[:num_to_process]
                        geometry = geometry[:num_to_process]
                        
                        src = encoder.encode_latents(r.src.to(device))
                        r.labels = geometry.to(device)
                    else: # Dual-source mode
                        base_src, r.tgt, geometry = (data[k][::2] for k in ["src_image", "tgt_image", "geometry"])
                        
                        num_to_process = min(len(r.seeds), base_src.shape[0])
                        if num_to_process == 0: continue
                        r.seeds = r.seeds[:num_to_process]
                        r.src = base_src[:num_to_process]
                        r.tgt = r.tgt[:num_to_process]
                        geometry = geometry[:num_to_process]

                        src_for_model = r.src.repeat_interleave(2, dim=0)
                        labels_for_model = geometry.repeat_interleave(2, dim=0)
                        src = encoder.encode_latents(src_for_model.to(device))
                        r.labels = labels_for_model.to(device)

                    rnd = StackedRandomGenerator(device, r.seeds)
                    if not VANILLA_MODE:
                        num_seeds = len(r.seeds)
                        noise_shape_per_seed = list(src.shape[1:])
                        noise_for_pairs = rnd.randn([num_seeds] + noise_shape_per_seed, device=device)
                        r.noise = noise_for_pairs.repeat_interleave(2, dim=0)
                    else:
                        r.noise = rnd.randn(src.shape, device=device)

                    src_for_depth = r.src if not super_res else data["sr_src_image"][:num_to_process]
                    if not VANILLA_MODE: src_for_depth = src_for_depth.repeat_interleave(2, dim=0)
                    src = add_depth(depth_model, src_for_depth.to(device), src, inv_norm=net.depth_input) if depth_model is not None else src
                    
                    if super_res:
                        tgt_for_sr = encoder.encode_latents(r.tgt.to(device))
                        low_res = torchvision.transforms.functional.resize(
                            torchvision.transforms.functional.resize(tgt_for_sr, tgt_for_sr.shape[-1] // 4),
                            tgt_for_sr.shape[-1]
                        )
                        sampler_kwargs['conditioning_image'] = low_res

                    with torch.no_grad():
                        latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, src=src, noise=r.noise,
                            labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                        r.images = encoder.decode(latents)
                    
                    if sr_model is not None:
                        r.src, r.tgt, sr_geometry = (data["sr_" + k] for k in ["src_image", "tgt_image", "geometry"])
                        
                        # Synchronize SR data with the number of processed seeds
                        r.src = r.src[:num_to_process]
                        r.tgt = r.tgt[:num_to_process]
                        sr_geometry = sr_geometry[:num_to_process]

                        sr_src = encoder.encode_latents(r.src.to(device))
                        rnd = StackedRandomGenerator(device, r.seeds)
                        r.noise = rnd.randn([len(r.seeds), sr_model.img_channels, sr_model.img_resolution, sr_model.img_resolution], device=device)
                        r.labels = sr_geometry.to(device)
                        low_res = torchvision.transforms.functional.resize(latents, sr_src.shape[-1])
                        sr_sampler_kwargs['conditioning_image'] = low_res
                        with torch.no_grad():
                            sr_latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=sr_model, src=sr_src, noise=r.noise,
                                labels=r.labels, gnet=sr_model, randn_like=rnd.randn_like, **sr_sampler_kwargs)
                            r.images = encoder.decode(sr_latents)

                    if outdir is not None:
                        for seed, _src, _tgt, image in zip(r.seeds, 
                                                            r.src.clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy(), 
                                                            r.tgt.clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy(), 
                                                            r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(_src, 'RGB').save(os.path.join(image_dir, f'src_{seed:06d}.png'))
                            PIL.Image.fromarray(_tgt, 'RGB').save(os.path.join(image_dir, f'tgt_{seed:06d}.png'))
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'sample_{seed:06d}.png'))

                torch.distributed.barrier()
                yield r

    return ImageIterable()


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

@click.option('--sr-model',                 help='Path to SR model to use for evaluation',metavar='STR',            type=str, default=None, show_default=True)
@click.option('--gnet',                     help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--guidance',                 help='Guidance factor',metavar='FLOAT',                                 type=float, default=None, show_default=True)
@click.option('--range-selection',          help='Range selection',metavar='MID,LONG',                              type=str, default=None, show_default=True)
@click.option('--depth-model',              help='Depth model to use for evaluation',metavar='STR',                 type=str, default=None, show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=vivid --outdir=out

    \b
    # Generate 10000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=vivid --outdir=out --subdirs --seeds=0-9999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images_nvs(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------