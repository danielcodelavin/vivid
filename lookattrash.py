import os
import shutil
import random
import torch
import litdata as ld
import torchvision.transforms.functional as TF
from PIL import Image

# --- Hardcoded Configuration ---
# ⚠️
# ⚠️  1. SET THIS TO THE PATH OF YOUR LITDATA STREAMING DATASET
# ⚠️  (e.g., the folder containing 'cache.json', 'index.json', etc.)
# ⚠️
DATA_PATH = "/storage/slurm/lavingal/lavingal/LVSM/datasets/gso_chunked"

# Folder to save the output images
OUTPUT_DIR = "trashoutput"

# How many different scenes/objects to load
NUM_SAMPLES_TO_VIEW = 5

# How many random images (views) to save from each sample
NUM_VIEWS_PER_SAMPLE = 4

# The target resolution for the output images
TARGET_IMAGE_SIZE = 256
# --- End of Configuration ---


def setup_output_directory():
    """Create a clean directory for the output images."""
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Removing existing output directory: '{OUTPUT_DIR}'")
        shutil.rmtree(OUTPUT_DIR)
    print(f"📂 Creating new output directory: '{OUTPUT_DIR}'")
    os.makedirs(OUTPUT_DIR)


def process_and_save_samples():
    """
    Loads data using litdata, processes a few samples, and saves visual outputs.
    """
    print(f"⚡ Initializing LitData streaming dataset from: '{DATA_PATH}'")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Data path not found: '{DATA_PATH}'")
        print("👉 Please update the 'DATA_PATH' variable in the script.")
        return

    # 1. Initialize the streaming dataset
    try:
        dataset = ld.StreamingDataset(input_dir=DATA_PATH)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize LitData dataset. Is the path correct? Details: {e}")
        return
        
    print(f"✅ Dataset initialized. Found {len(dataset)} items.")
    print("-" * 30)

    processed_count = 0
    # 2. Iterate through the dataset to get a few samples
    for i, sample in enumerate(dataset):
        if processed_count >= NUM_SAMPLES_TO_VIEW:
            print(f"\n✅ Finished processing {processed_count} samples.")
            break

        try:
            # --- This section mimics the logic from your LitDataCollate ---

            # a. Validate the sample
            if not sample or 'image' not in sample or sample['image'].shape[0] < NUM_VIEWS_PER_SAMPLE:
                print(f"⏭️  Skipping sample {i} (invalid or not enough views).")
                continue
            
            scene_name = sample.get('scene_name', f'unknown_scene_{i}')
            images = sample['image'] # Shape: (num_available_views, C, H, W)
            num_available_views = images.shape[0]

            print(f"🔎 Processing sample {i}: '{scene_name}' (has {num_available_views} views)")

            # b. Select a random subset of views
            image_indices = random.sample(range(num_available_views), NUM_VIEWS_PER_SAMPLE)
            print(f"   - Randomly selected view indices: {image_indices}")

            # c. Resize, convert, and save each selected view
            for view_idx in image_indices:
                img_tensor = images[view_idx]

                # Resize using the same function as your collate script
                resized_tensor = TF.resize(
                    img_tensor, 
                    [TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE], 
                    antialias=True
                )

                # Convert from PyTorch tensor (C, H, W) to a PIL Image (H, W, C)
                # We assume the tensor is in [0, 1] float format.
                # 1. Permute from (C, H, W) to (H, W, C)
                # 2. Scale from [0, 1] to [0, 255]
                # 3. Convert to a NumPy array of unsigned 8-bit integers
                # 4. Create a PIL Image from the array
                img_np = (resized_tensor.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
                pil_image = Image.fromarray(img_np)

                # Save the image to the output folder
                output_filename = f"{scene_name}_view_{view_idx}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                pil_image.save(output_path)
                print(f"   💾 Saved resized image to: '{output_path}'")

            processed_count += 1
            print("-" * 30)

        except Exception as e:
            print(f"⚠️  WARNING: Could not process sample {i}. Error: {e}")
            continue

if __name__ == "__main__":
    setup_output_directory()
    process_and_save_samples()
    print("\n🎉 Script finished. Check the 'trashoutput' folder for your images.")