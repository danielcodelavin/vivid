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

"""Main training loop."""

import datetime
import os
import time
import copy
import pickle
import PIL
import psutil
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms.functional
from torch.utils.tensorboard import SummaryWriter
import dnnlib
from generate_images import edm_sampler
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from training.utils import add_depth, compose_geometry, open_tensorboard_process, resolve_model, resolve_depth_model
from calculate_metrics import get_metrics

    

@persistence.persistent_class
class NVLoss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, src, tgt, labels=None):
        rnd_normal = torch.randn([tgt.shape[0], 1, 1, 1], device=tgt.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(tgt) * sigma
        denoised, logvar = net(src, tgt + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - tgt) ** 2) + logvar
        return loss


@persistence.persistent_class
class SRNVLoss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, src, tgt, labels=None):
        rnd_normal = torch.randn([tgt.shape[0], 1, 1, 1], device=tgt.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(tgt) * sigma
        low_res = torchvision.transforms.functional.resize(
            torchvision.transforms.functional.resize(tgt, tgt.shape[-1] // 4),
            tgt.shape[-1]
        )
        denoised, logvar = net(src, tgt + noise, sigma, labels, conditioning_image=low_res, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - tgt) ** 2) + logvar
        return loss


#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='datautils.RealEstate10K'),
    encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=4, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.NVDiffLoss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    samples_nimg        = 1024<<10,  # Report status every N training images. None = disable.
    metrics_nimg        = 1024<<10,  # Metric status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),

    eval_samples        = 8,       # Number of samples to generate for logging. 0 = no generated samples
    sr_training         = False,    # Toggle to train an SR model
    single_image_mix    = None,     # Percent of per-gpu batch to train with single image augmentation. None = only multiview data
    sr_model            = None,     # SR model to use for logging and metrics. None = log only current res
    depth_model         = None,     # Depth model size to use for training
    debug               = None,     # Disable logging
):
    params = locals()
    params.pop('device')
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if dist.get_rank() == 0 and debug is None:
        writer = SummaryWriter(log_dir=run_dir)
        open_tensorboard_process(run_dir)


    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert metrics_nimg is None or (metrics_nimg % batch_size == 0 and metrics_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    if single_image_mix is not None and single_image_mix > 0:
        assert single_image_mix % batch_gpu > 0, "cant use less than 1 single image per GPU"
        from datautils.openimages import get_openimages_dataloader
        single_image_mix = min(batch_gpu - 1, int(batch_gpu * single_image_mix))
        single_image_dataset = dnnlib.util.construct_class_by_name(class_name='datautils.SingleImages', **{k: v for k, v in dataset_kwargs.items() if k != "class_name"})
    ref_image, _, ref_label = (dataset_obj[0][("sr_" if sr_training else "") + k] for k in ["src_image", "tgt_image", "geometry"])
    ref_label = ref_label.reshape((-1,))
    test_dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k != "split"}
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels + int(depth_model is not None), net.img_resolution, net.img_resolution], device=device),
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ] + ([torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)] if net.super_res else []), 
        max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Init extra models
    sr_model = resolve_model(sr_model, name="SR")
    depth_model = resolve_depth_model(depth_model, device=device)

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg ({(stop_at_nimg - state.cur_nimg) // batch_size} iters):')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, 
                                                                batch_size=(batch_gpu - (single_image_mix if single_image_mix is not None else 0)), **data_loader_kwargs))
    if single_image_mix:
        single_image_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
        single_image_iterator = iter(dnnlib.util.construct_class_by_name(dataset=single_image_dataset, sampler=single_images_sampler,  batch_size=single_image_mix, **data_loader_kwargs))
    if dist.get_rank() == 0 and eval_samples:
        test_dataset_obj = dnnlib.util.construct_class_by_name(split="test", **test_dataset_kwargs)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset_obj, batch_size=eval_samples, shuffle=False, num_workers=1, drop_last=False, pin_memory=False, persistent_workers=True)  
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'iter',         f"{training_stats.report0('Progress/iter',                              state.cur_nimg / batch_size):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0 and debug is None:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                for name, value in items:
                    writer.add_scalar(name, value, state.cur_nimg // batch_size)
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

                # Generate sample images
                if eval_samples and samples_nimg is not None and (done or state.cur_nimg % samples_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
                    with torch.no_grad():
                        net.eval()
                        data = next(iter(test_dataset_loader))
                        src_image, tgt_image, geometry = (data[("sr_" if sr_training else "") + k] for k in ["src_image", "tgt_image", "geometry"])
                        src = encoder.encode_latents(src_image.to(device))
                        tgt = encoder.encode_latents(tgt_image.to(device))
                        sampler_kwargs = {}
                        src = add_depth(depth_model, data["sr_src_image"].to(device), src, inv_norm=net.depth_input) if depth_model is not None else src
                        if net.super_res:
                            low_res = torchvision.transforms.functional.resize(
                                torchvision.transforms.functional.resize(tgt, tgt.shape[-1] // 4),
                                tgt.shape[-1]
                            )
                            sampler_kwargs['conditioning_image'] = low_res
                        latents = edm_sampler(net=net, src=src, noise=torch.randn_like(tgt), labels=geometry.to(device), **sampler_kwargs)
                        net.train()
                        predicted_images = encoder.decode(latents).cpu()
                        if net.super_res:
                            lr_image = encoder.decode(low_res).cpu()
                            samples_grid = torchvision.utils.make_grid(torch.cat([lr_image, src_image, predicted_images, tgt_image], dim=0), eval_samples).to(torch.uint8).cpu()
                        else:
                            samples_grid = torchvision.utils.make_grid(torch.cat([src_image, predicted_images, tgt_image], dim=0), eval_samples).to(torch.uint8).cpu()

                        # Save images localy
                        PIL.Image.fromarray(
                            samples_grid.permute(1, 2, 0).numpy()
                        ).save(os.path.join(run_dir, "results", f'generated-samples-{state.cur_nimg//1000:07d}.png'))
                        writer.add_image("Image/samples", samples_grid, state.cur_nimg // batch_size)
                        writer.flush()

                        # Up sample with SR model if exists
                        if sr_model is not None:
                            sr_src_image, sr_tgt_image, sr_geometry = (data["sr_" + k] for k in ["src_image", "tgt_image", "geometry"])
                            sr_src = encoder.encode_latents(sr_src_image.to(device))
                            low_res = torchvision.transforms.functional.resize(latents, sr_src.shape[-1])
                            sampler_kwargs['conditioning_image'] = low_res
                            sr_latents = edm_sampler(net=sr_model, src=sr_src, noise=torch.randn_like(sr_src), labels=sr_geometry.to(device), **sampler_kwargs)
                            predicted_sr_images = encoder.decode(sr_latents).cpu()
                            sr_samples_grid = torchvision.utils.make_grid(torch.cat([sr_src_image, predicted_sr_images, sr_tgt_image], dim=0), eval_samples).to(torch.uint8).cpu()
                            PIL.Image.fromarray(
                                sr_samples_grid.permute(1, 2, 0).numpy()
                            ).save(os.path.join(run_dir, "results", f'generated-sr-samples-{state.cur_nimg//1000:07d}.png'))
                            writer.add_image("Image/sr-samples", sr_samples_grid, state.cur_nimg // batch_size)  
                            writer.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True
        
        # Compute metrics
        if metrics_nimg is not None and (done or state.cur_nimg % metrics_nimg == 0) and (state.cur_nimg != start_nimg):
            net.eval()
            metrics_mid = get_metrics(net=net, encoder=encoder, sr_model=sr_model, depth_model=depth_model, datakwargs=dict(**test_dataset_kwargs,range_selection="mid"))
            metrics_long = get_metrics(net=net, encoder=encoder, sr_model=sr_model, depth_model=depth_model, datakwargs=dict(**test_dataset_kwargs,range_selection="long"))
            net.train()
            if dist.get_rank() == 0:
                for name, value in {f"Metrics/mid_{k}": v for k, v in metrics_mid.items()}.items():
                    writer.add_scalar(name, value, state.cur_nimg // batch_size)
                for name, value in {f"Metrics/long_{k}": v for k, v in metrics_long.items()}.items():
                    writer.add_scalar(name, value, state.cur_nimg // batch_size)
                writer.flush()

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            fname = f'training-state-{state.cur_nimg//1000:07d}.pt'
            checkpoint.save(os.path.join(run_dir, fname))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            if dist.get_rank() == 0 and debug is None:
                writer.close()
            break

        torch.distributed.barrier()

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data = next(dataset_iterator)
                src_image, tgt_image, geometry = (data[("sr_" if sr_training else "") + k] for k in ["src_image", "tgt_image", "geometry"])
                high_res = data["sr_src_image"]

                # Mix single image data
                if single_image_mix is not None:
                    single_image_data  = next(single_image_loader)
                    _src_image, _tgt_image, _geometry = (single_image_data[("sr_" if sr_training else "") + k] for k in ["src_image", "tgt_image", "geometry"])
                    _high_res = single_image_data["sr_src_image"]
                    src_image, tgt_image, geometry, high_res = (torch.cat([multi_view, single_image], 0) for multi_view, single_image in zip([src_image, tgt_image, geometry, high_res], [_src_image, _tgt_image, _geometry, _high_res]))
                    assert src_image.shape[0] == batch_gpu
                
                src_image = encoder.encode_latents(src_image.to(device))
                tgt_image = encoder.encode_latents(tgt_image.to(device))
                geometry = geometry.to(device)
                src_image = add_depth(depth_model, high_res.to(device), src_image, inv_norm=net.depth_input) if depth_model is not None else src_image
                loss = loss_fn(net=ddp, src=src_image, tgt=tgt_image, labels=geometry)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------
