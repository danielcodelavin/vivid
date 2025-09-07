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
import random
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from training.utils import add_depth, compose_geometry, resolve_model, resolve_depth_model
from calculate_metrics import get_metrics
from .custom_litdata_loader import CustomLitCollate , CustomLitDataset, VANILLA_MODE
from tqdm import tqdm
import torch.cuda
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from fvcore.nn import FlopCountAnalysis


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

        if not VANILLA_MODE:
            # In dual-source mode, `denoised` is [B,...] but `tgt` and `weight` are [2*B,...]
            # We must slice tgt and weight to match the model's output.
            loss = (weight[::2] / logvar.exp()) * ((denoised - tgt[::2]) ** 2) + logvar
        else:
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

        if not VANILLA_MODE:
            loss = (weight[::2] / logvar.exp()) * ((denoised - tgt[::2]) ** 2) + logvar
        else:
            loss = (weight / logvar.exp()) * ((denoised - tgt) ** 2) + logvar
        return loss

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

def analyze_flops(net, device, batch_size, img_resolution, img_channels, source_label_dim,target_label_dim, sr_training):
    if dist.get_rank() != 0:
        return

    print("--- Starting GFLOPs Analysis with Deepspeed FlopsProfiler ---")
    model_for_flops = (net.module if isinstance(net, DDP) else net).to(device)
    model_for_flops.eval()
    
    prof = None
    try:
        from deepspeed.profiling.flops_profiler import FlopsProfiler

        # Generate dummy tensors consistent with the selected data loading mode.
        if VANILLA_MODE:
            dummy_bs = batch_size
            dummy_src = torch.randn(dummy_bs, img_channels, img_resolution, img_resolution, dtype=torch.float32, device=device)
            dummy_dst = torch.randn(dummy_bs, img_channels, img_resolution, img_resolution, dtype=torch.float32, device=device)
            dummy_sigma = torch.randn(dummy_bs, device=device)
            dummy_geometry = torch.randn(dummy_bs, source_label_dim, device=device)
        else: # Dual-source mode
            dummy_bs = batch_size * 2
            dummy_src = torch.randn(dummy_bs, img_channels, img_resolution, img_resolution, dtype=torch.float32, device=device)
            dummy_dst = torch.randn(dummy_bs, img_channels, img_resolution, img_resolution, dtype=torch.float32, device=device)
            dummy_sigma = torch.randn(dummy_bs, device=device)
            dummy_geometry = torch.randn(dummy_bs, source_label_dim, device=device)
        
        conditioning_image = None
        if sr_training:
            cond_bs = batch_size
            conditioning_image = torch.randn(cond_bs, img_channels, img_resolution, img_resolution, dtype=torch.float32, device=device)

        prof = FlopsProfiler(model_for_flops)
        prof.start_profile()

        _ = model_for_flops(src=dummy_src, dst=dummy_dst, sigma=dummy_sigma, geometry=dummy_geometry, conditioning_image=conditioning_image)

        flops = prof.get_total_flops()
        gflops = flops / 1e9
        
        print(f" GFLOPs per forward pass: {gflops:.2f} G")

        if wandb.run:
            wandb.summary["GFLOPs"] = gflops
    
    except Exception as e:
        import traceback
        print(f"Could not complete FLOPs analysis due to an error: {e}")
        traceback.print_exc()
        
    finally:
        if prof is not None:
            prof.end_profile() 
        model_for_flops.train()
        print("--- GFLOPs Analysis Complete ---")

def training_loop(
    dataset_kwargs      = dict(class_name='training.datautils.custom_litdata_loader.CustomLitDataset', path='/storage/user/lavingal/re10k_train_chunks_all_views'),
    test_dataset_path   = '/storage/user/lavingal/re10k_test_chunks_all_views', 
    encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=8, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.NVDiffLoss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',
    seed                = 0,
    batch_size          = 512,
    batch_gpu           = None,
    total_nimg          = 8<<30,
    slice_nimg          = None,
    status_nimg         = 80,
    samples_nimg        = 10000,
    metrics_nimg        = 10000,
    snapshot_nimg       = 10000,
    checkpoint_nimg     = 10000,

    loss_scaling        = 1,
    force_finite        = True,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),

    eval_samples        = 8,
    sr_training         = False,
    single_image_mix    = None,
    sr_model            = None,
    depth_model         = None,
    debug               = None,
):
    params = locals()
    params.pop('device')

    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    dist.print0('Initializing custom LitData pipeline...')
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    
    train_cache_dir = f"/dev/shm/litdata_cache_{job_id}_train"
    test_cache_dir = f"/dev/shm/litdata_cache_{job_id}_test"
    if dist.get_rank() == 0:
        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(test_cache_dir, exist_ok=True)
    dist.barrier() 
    
    mode_str = 'vanilla' if VANILLA_MODE else 'dual_source'
    
    # Pass the explicit instruction when creating the dataset
    dataset_obj = CustomLitDataset(
        path=dataset_kwargs['path'],
        cache_dir=train_cache_dir,
        mode=mode_str
    )

    if single_image_mix is not None and single_image_mix > 0:
        assert single_image_mix % batch_gpu > 0, "cant use less than 1 single image per GPU"
        from datautils.openimages import get_openimages_dataloader
        single_image_mix = min(batch_gpu - 1, int(batch_gpu * single_image_mix))
        single_image_dataset = dnnlib.util.construct_class_by_name(class_name='datautils.SingleImages', **{k: v for k, v in dataset_kwargs.items() if k != "class_name"})

    target_resolution = 256 if sr_training else 64
    dist.print0(f'Setting up for a {target_resolution}x{target_resolution} resolution model...')

    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    
    source_label_dim = 20
    if VANILLA_MODE:
        target_label_dim = source_label_dim
    else:
        target_label_dim = source_label_dim * 2
    dist.print0(f'Constructing network with resolution={target_resolution}x{target_resolution}...')
    interface_kwargs = dict(img_resolution=target_resolution, img_channels=3, source_label_dim=source_label_dim, target_label_dim=target_label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    sr_model = resolve_model(sr_model, name="SR")
    depth_model = resolve_depth_model(depth_model, device=device)

    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg ({(stop_at_nimg - state.cur_nimg) // batch_size} iters):')
    dist.print0()

    collate_fn_instance = CustomLitCollate()
    data_loader_kwargs['collate_fn'] = collate_fn_instance
    data_loader_kwargs.pop('class_name', None)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_obj, 
        sampler=None,
        batch_size=batch_gpu if VANILLA_MODE else batch_gpu * 2,
        **data_loader_kwargs
    )
    dataset_iterator = iter(dataloader)
    
    if single_image_mix:
        single_image_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
        single_image_iterator = iter(dnnlib.util.construct_class_by_name(dataset=single_image_dataset, sampler=single_image_sampler,  batch_size=single_image_mix, **data_loader_kwargs))

    if dist.get_rank() == 0 and eval_samples:
        dist.print0('Setting up test dataloader for sample generation...')
        test_dataset_obj = CustomLitDataset(path=test_dataset_path, cache_dir=test_cache_dir)
        test_dataset_loader = torch.utils.data.DataLoader(
            test_dataset_obj,
            batch_size=eval_samples if VANILLA_MODE else eval_samples * 2,
            collate_fn=collate_fn_instance,
            shuffle=False,
            num_workers=data_loader_kwargs['num_workers'],
            pin_memory=True
        )
        test_dataset_iterator = iter(test_dataset_loader)

    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    start_iter = state.cur_nimg // batch_size
    total_iters = stop_at_nimg // batch_size
    lr = 0.0

    if dist.get_rank() == 0:
        analyze_flops(
            net=ddp,
            device=device,
            batch_size=batch_gpu,
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            source_label_dim=source_label_dim,
            target_label_dim= target_label_dim,
            sr_training=sr_training
        )
    dist.barrier()

    progress_bar = tqdm(range(start_iter, total_iters), initial=start_iter, total=total_iters, unit="iter", desc="Training", leave=True)
    for _ in progress_bar:
        done = (state.cur_nimg >= stop_at_nimg)
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg', state.cur_nimg / 1e3):<9.1f}",
                'iter',         f"{training_stats.report0('Progress/iter', state.cur_nimg / batch_size):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec', state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick', cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg', cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec', cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb', cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            training_stats.default_collector.update()
            if dist.get_rank() == 0 and debug is None:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()
                if wandb.run:
                    stats_dict = training_stats.default_collector.as_dict()
                    loss_stat = stats_dict.get('Loss/loss')
                    log_dict = {"learning_rate": lr , "kimg": state.cur_nimg / 1e3}
                    if loss_stat is not None:
                        log_dict["loss"] = loss_stat.mean   
                    for name, value in stats_dict.items():
                        log_dict[name.replace('/', '_')] = value.mean
                    wandb.log(log_dict, step=state.cur_nimg)

                if eval_samples and samples_nimg is not None and (done or state.cur_nimg % samples_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
                    with torch.no_grad():
                        net.eval()
                        try:
                            data = next(test_dataset_iterator)
                        except StopIteration:
                            dist.print0('Resetting test dataloader iterator...')
                            test_dataset_iterator = iter(test_dataset_loader)
                            data = next(test_dataset_iterator)
                        
                       
                        src_image, tgt_image, geometry = (data[("sr_" if sr_training else "") + k] for k in ["src_image", "tgt_image", "geometry"])
                        
                        if not VANILLA_MODE:
                            # We only need one source and its geometry for sampling visualization.
                            src_image = src_image[::2]
                            tgt_image = tgt_image[::2]
                            geometry = geometry[::2]
                        
                        src = encoder.encode_latents(src_image.to(device))
                        tgt = encoder.encode_latents(tgt_image.to(device))
                        sampler_kwargs = {}
                        
                        src_for_depth = src_image if not sr_training else data["sr_src_image"]
                        src = add_depth(depth_model, src_for_depth.to(device), src, inv_norm=net.depth_input) if depth_model is not None else src
                    # up until here target correctly the same
                    #    src = add_depth(depth_model, src_for_depth.to(device), src, inv_norm=net.depth_input) if depth_model is not None else src
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

                        PIL.Image.fromarray(
                            samples_grid.permute(1, 2, 0).numpy()
                        ).save(os.path.join(run_dir, "results", f'generated-samples-{state.cur_nimg//1000:07d}.png'))

                        if dist.get_rank() == 0 and wandb.run:
                            wandb.log({"Generated Samples": wandb.Image(samples_grid)}, step=state.cur_nimg)
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

            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True
        
        if metrics_nimg is not None and (done or state.cur_nimg % metrics_nimg == 0) and (state.cur_nimg != start_nimg):
            net.eval()
            if test_dataset_path is not None:
                test_dataset_kwargs = dict(dataset_kwargs)
                test_dataset_kwargs['path'] = test_dataset_path
            test_dataset_kwargs.pop('split', None)
            metrics = get_metrics(net=net, encoder=encoder, sr_model=sr_model, depth_model=depth_model, datakwargs=test_dataset_kwargs)
            net.train()
            if dist.get_rank() == 0 and wandb.run:
                wandb_metrics = {f"Metrics/{key}": value for key, value in metrics.items()}
                wandb.log(wandb_metrics, step=state.cur_nimg)
            
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
                del data

        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            fname = f'training-state-{state.cur_nimg//1000:07d}.pt'
            checkpoint.save(os.path.join(run_dir, fname))
            misc.check_ddp_consistency(net)

        if done:
            break

        torch.distributed.barrier()
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data = next(dataset_iterator, None)
                if data is None:
                    continue
               
                src_image = data['src_image']
                tgt_image = data['tgt_image']
                geometry = data['geometry']
                high_res = data.get('high_res_src_image', src_image)
                
                src_image = encoder.encode_latents(src_image.to(device))
                tgt_image = encoder.encode_latents(tgt_image.to(device))
                geometry = geometry.to(device)
                
                if depth_model is not None:
                    src_image = add_depth(depth_model, high_res.to(device), src_image, inv_norm=net.depth_input)
                
                
                if VANILLA_MODE:
                    loss = loss_fn(net=ddp, src=src_image, tgt=tgt_image, labels=geometry)
                else:
                    # 1. Calculate the number of image pairs
                    num_pairs = tgt_image.shape[0] // 2
                    
                    # 2. Generate one sigma value for each pair
                    rnd_normal = torch.randn([num_pairs, 1, 1, 1], device=tgt_image.device)
                    sigma_targets = (rnd_normal * loss_fn.P_std + loss_fn.P_mean).exp()
                    
                    # 3. Generate one unique noise tensor for each pair
                    noise_pairs = torch.randn([num_pairs, *tgt_image.shape[1:]], device=tgt_image.device) * sigma_targets

                    # 4. Duplicate the noise so that both images in a pair have the same noise
                    noise = torch.repeat_interleave(noise_pairs, repeats=2, dim=0)
                    
                    # Add the paired noise to the target images
                    noisy_targets_full = tgt_image + noise

                    # The rest of the logic remains the same
                    denoised, logvar = ddp(src=src_image, dst=noisy_targets_full, sigma=sigma_targets, geometry=geometry, return_logvar=True)
                    
                    weight = (sigma_targets ** 2 + loss_fn.sigma_data ** 2) / (sigma_targets * loss_fn.sigma_data) ** 2
                    loss = (weight / logvar.exp()) * ((denoised - tgt_image[::2]) ** 2) + logvar

                training_stats.report('Loss/loss', loss)
                progress_bar.set_postfix(loss=loss.mean().item())

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
        

        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

    dist.print0('Training done.')
    if dist.get_rank() == 0 and wandb.run:
        wandb.finish()