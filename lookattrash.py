import os
import shutil
import random
import torch
import litdata as ld
import torchvision.transforms.functional as TF
from PIL import Image

DATA_PATH = "/storage/user/lavingal/objaverseplus_chunked"
OUTPUT_DIR = "trashoutput"
NUM_SCENES = 30
TARGET_IMAGE_SIZE = 256


def setup_output_directory():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)


def process_and_save_samples():
    dataset = ld.StreamingDataset(input_dir=DATA_PATH)
    
    scene_indices = random.sample(range(len(dataset)), NUM_SCENES)
    
    for idx in scene_indices:
        sample = dataset[idx]
        
        scene_name = sample.get('scene_name', f'scene_{idx}')
        images = sample['image']
        num_views = images.shape[0]
        
        scene_folder = os.path.join(OUTPUT_DIR, scene_name)
        os.makedirs(scene_folder, exist_ok=True)
        
        for view_idx in range(num_views):
            img_tensor = images[view_idx]
            resized_tensor = TF.resize(img_tensor, [TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE], antialias=True)
            img_np = (resized_tensor.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img_np)
            
            output_path = os.path.join(scene_folder, f"view_{view_idx}.png")
            pil_image.save(output_path)


if __name__ == "__main__":
    setup_output_directory()
    process_and_save_samples()
    print("fin")