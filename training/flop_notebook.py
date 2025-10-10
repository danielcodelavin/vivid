# ======================================================================================
# FLOPs Analysis Script for Generative Model
#
# INSTRUCTIONS:
# 1. Save this file in your project's root directory.
# 2. Adjust the parameters in the CONFIGURATION section below.
# 3. Run from the terminal: `python training/flop_notebook.py`
# ======================================================================================

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dnnlib

try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
except ImportError as e:
    print("Import Error: Deepspeed not found. Please install it.")
    exit()

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# --- Analysis Mode ---
DUAL_SOURCE_MODE = True

# --- Hardware Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Architectural Parameters ---
BATCH_SIZE = 64
IMG_RESOLUTION = 64
IMG_CHANNELS = 3
MODEL_CHANNELS = 64
SOURCE_LABEL_DIM = 20
TARGET_LABEL_DIM = 40
SR_TRAINING = False
NUM_TARGETS = 1


# --- PROFILER SETTINGS ---
MODULE_DEPTH = 1  # -1 for full depth, 1 for top-level only, 2 for two levels, etc.
TOP_MODULES = 3   # Number of top modules to display in the summary.

#  U-Net Parameters 
CHANNEL_MULT = [1, 2, 3, 4]
NUM_BLOCKS = 3
ATTN_RESOLUTIONS = [16, 8]

## DEFAULTS FOR VIVID:
    #     img_channels,                       # Image channels.
    #     label_dim,                          # Class label dimensionality. 0 = unconditional.
    #     model_channels      = 192,          # Base multiplier for the number of channels.
    #     channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
    #     channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
    #     channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
    #     num_blocks          = 3,            # Number of residual blocks per resolution.
    #     attn_resolutions    = [16,8],       # List of resolutions with self-attention.
    #     label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
    #     concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
    #     extra_attn          = None,         # Force attention at the start and end of every level
    #     epipolar_attention_bias          = False,
    #     **block_kwargs,               




# --- Network Kwargs ---
NETWORK_KWARGS = dict(
    class_name='training.models.NVPrecond',
    extra_attn=1,
)

# ======================================================================================
# ANALYSIS
# ======================================================================================

def run_flops_analysis(config):
    """
    Sets up the model, generates dummy data, and runs the FLOPs analysis.
    """
    # 1. Construct the Model
    print("--- 1. Constructing Model ---")
    interface_kwargs = dict(
        img_resolution=config['img_resolution'],
        img_channels=config['img_channels'],
        source_label_dim=config['source_label_dim'],
        target_label_dim=config['target_label_dim'],
    )

    network_kwargs = config['network_kwargs'].copy()
    network_kwargs['model_channels'] = config['model_channels']
    
    # --- ADD THE NEW U-NET PARAMS TO THE MODEL CONFIG ---
    network_kwargs['channel_mult'] = config['channel_mult']
    network_kwargs['num_blocks'] = config['num_blocks']
    network_kwargs['attn_resolutions'] = config['attn_resolutions']

    model = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    model.eval().to(config['device'])
    print(f"Model constructed successfully and moved to '{config['device']}'.")

    # 2. Generate Dummy Data
    print("\n--- 2. Generating Dummy Input Tensors ---")
    base_bs = config['batch_size']
    effective_bs = base_bs * 2 * NUM_TARGETS
    
    print(f"{'Dual-source' if config['dual_source_mode'] else 'Vanilla'} mode enabled. Effective tensor batch size: {effective_bs}")
        
    dummy_src = torch.randn(effective_bs, config['img_channels'], config['img_resolution'], config['img_resolution'], device=config['device'])
    dummy_dst = torch.randn(effective_bs, config['img_channels'], config['img_resolution'], config['img_resolution'], device=config['device'])
    dummy_sigma = torch.randn(effective_bs, device=config['device'])
    dummy_geometry = torch.randn(effective_bs, config['source_label_dim'], device=config['device'])
    
    conditioning_image = None
    if config['sr_training']:
        conditioning_image = torch.randn(base_bs, config['img_channels'], config['img_resolution'], config['img_resolution'], device=config['device'])
        
    dummy_inputs = {
        "src": dummy_src, "dst": dummy_dst, "sigma": dummy_sigma,
        "geometry": dummy_geometry, "conditioning_image": conditioning_image
    }
    print("Dummy tensors generated with the following shapes:")
    print(f"  - src/dst shape: {dummy_src.shape}")
    print(f"  - geometry shape: {dummy_geometry.shape} (Matches UNetEncoder's expectation)")

    # 3. Run Deepspeed FlopsProfiler
    print("\n--- 3. Analyzing FLOPs with Deepspeed ---")
    prof = FlopsProfiler(model)
    prof.start_profile()
    
    with torch.no_grad():
        _ = model(**dummy_inputs)
        
    print("\n--- Detailed FLOPs Breakdown per Module ---")
    prof.print_model_profile(profile_step=0, module_depth=MODULE_DEPTH, top_modules=TOP_MODULES, detailed=True, output_file=None)


    
    flops = prof.get_total_flops()
    params = prof.get_total_params()
    prof.end_profile()
    
    

    print("\n--- Total FLOPs Summary ---")
    gflops = flops / 1e9
    print(f"  - GFLOPs: {gflops:.4f} G")
    print(f"  - Total Parameters: {params / 1e6:.2f} M")

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================

if __name__ == "__main__":
    
    config = {
        "dual_source_mode": DUAL_SOURCE_MODE,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "img_resolution": IMG_RESOLUTION,
        "img_channels": IMG_CHANNELS,
        "model_channels": MODEL_CHANNELS,
        "source_label_dim": SOURCE_LABEL_DIM,
        "target_label_dim": TARGET_LABEL_DIM,
        "sr_training": SR_TRAINING,
        "network_kwargs": NETWORK_KWARGS,
        # --- ADD NEW PARAMS TO THE CONFIG DICTIONARY ---
        "channel_mult": CHANNEL_MULT,
        "num_blocks": NUM_BLOCKS,
        "attn_resolutions": ATTN_RESOLUTIONS,
    }
    
    print("=" * 60)
    print("Starting FLOPs Analysis with Configuration:")
    for key, val in config.items():
        if isinstance(val, dict): continue
        print(f"  - {key}: {val}")
    print("=" * 60)

    run_flops_analysis(config)

    print("\n--- Analysis Complete ---")