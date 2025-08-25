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

import datetime
import os
import re
import json
import shutil
from glob import glob
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'vivid-base':    dnnlib.EasyDict(duration=1024<<20, batch=1024,  channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6, extra_attn=1),
    'vivid-uncond':    dnnlib.EasyDict(duration=1024<<19, batch=1024,  channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6, extra_attn=1, uncond=True),
    'vivid-sr':   dnnlib.EasyDict(duration=256<<20,  batch=128,  channels=64,  lr=0.0200, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6, noisy_sr=0.25),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='vivid-base', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)


    # Dataset.
    path_to_dataset = '/storage/user/lavingal/re10k_train_chunks_all_views'
    c.dataset_kwargs = dnnlib.EasyDict(class_name='datautils.RealEstate10K', path=path_to_dataset, split="train", imsize=64)

    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='training.models.NVPrecond', model_channels=opts.channels, dropout=opts.dropout, 
                                       extra_attn=opts.extra_attn, epipolar_attention_bias=opts.epipolar_attn_bias,
                                       super_res=opts.sr_training, no_time_enc=opts.no_time_enc,
                                       depth_input=opts.depth_input, warp_depth_coor=opts.warp_depth_coor,
                                       uncond=opts.uncond, noisy_sr=opts.noisy_sr)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.' + ('NVLoss' if not opts.sr_training else 'SRNVLoss'), P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.samples_nimg = opts.get('samples', 0) or None
    c.metrics_nimg = opts.get('metrics', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)
    c.debug = opts.get('debug', 0) or None
    c.sr_model = opts.get('sr_model', 0) or None
    c.depth_model = opts.get('depth_model', 0) or None
    c.single_image_mix = opts.get('single_image_mix', 0) or None
    c.sr_training = opts.get('sr_training', 0) or False
    c.test_dataset_path = opts.get('test_data_path', None)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_local_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        
    if dist.get_local_rank() == 0 and not os.path.isdir(os.path.join(run_dir, "results")):
        os.makedirs(os.path.join(run_dir, "results"))
        code_dir = os.path.join(run_dir, "code")
        os.makedirs(code_dir)
        python_files = glob("./**[!experiments, !data]/*.py", recursive=True) + glob("./*.py", recursive=True)
        for path in python_files:
            os.makedirs(os.path.dirname(path.replace("./", code_dir + os.path.sep)), exist_ok=True)
            shutil.copy(path, path.replace("./", code_dir + os.path.sep))

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, default='output/')
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='vivid-base', show_default=True)
@click.option('--sr-training',      help='Toggles training of SR model',                        is_flag=False)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# NVS params
@click.option('--epipolar-attn-bias',help='Use epipolar attn bias', metavar='BOOL',             type=bool, default=False, show_default=True)
@click.option('--no-time-enc',      help='Nullify time input in Encoder model',                 is_flag=True)
@click.option('--depth-model',      help='Depth model type', metavar='small|base|large',        type=str, default=None, show_default=True)
@click.option('--depth-input',      help='Adds depth in input',                                 is_flag=True)
@click.option('--warp-depth-coor',  help='Add coordinates and warped coordinates as input',     is_flag=True)
@click.option('--single-image-mix', help='Use single image augmentations, percent of batch',    type=float, default=None, show_default=True)
@click.option('--uncond',           help='Regular diffusion',                                   is_flag=True)
@click.option('--noisy-sr',         help='Adds noise to low-res image',                         type=float, default=None, show_default=True)
@click.option('--sr-model',         help='Path to SR model to use for evaluation',metavar='STR',type=str, required=False)
@click.option('--test-data-path',   help='Path to the test dataset chunks', metavar='DIR', type=str, default='/storage/user/lavingal/re10k_test_chunks_all_views')

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=2, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--samples',          help='Interval of sample generation', metavar='NIMG',       type=parse_nimg, default='1024Ki', show_default=True)
@click.option('--metrics',          help='Interval of metrics prints', metavar='NIMG',          type=parse_nimg, default='8Mi', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='16Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--dry-run',          help='Print training options and exit',                     is_flag=True)


def cmdline(outdir, dry_run, **opts):
   # torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    outdir = os.path.join(outdir, "experiments")
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    cmdline()

