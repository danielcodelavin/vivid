import torch
import litdata as ld
import random
import torchvision.transforms.functional as F
from training.utils import compose_geometry

# This global variable will be used in the training loop to select the correct collate function.
VANILLA_MODE = True


class VanillaCollate:
    def __call__(self, batch: list[dict]) -> dict | None:
        valid_scenes = [s for s in batch if s and 'image' in s and s['image'].shape[0] >= 2]
        if not valid_scenes:
            return None

        src_images, tgt_images, geometries = [], [], []

        for scene in valid_scenes:
            try:
                num_available = scene['image'].shape[0]
                idx1, idx2 = random.sample(range(num_available), 2)
               
                src_img_raw = scene['image'][idx1]
                tgt_img_raw = scene['image'][idx2]

                src_img_float = src_img_raw.float()
                tgt_img_float = tgt_img_raw.float()

                
                if src_img_float.max() < 2.0:
                    src_img_float = src_img_float * 255.0
                if tgt_img_float.max() < 2.0:
                    tgt_img_float = tgt_img_float * 255.0
                
                
                src_img_resized = F.resize(src_img_float, [64, 64], antialias=True)
                tgt_img_resized = F.resize(tgt_img_float, [64, 64], antialias=True)
               

                src_c2w = torch.as_tensor(scene['c2w'][idx1], dtype=torch.float32)
                tgt_c2w = torch.as_tensor(scene['c2w'][idx2], dtype=torch.float32)

                tgt2src = torch.linalg.inv(tgt_c2w) @ src_c2w
                geo = compose_geometry(
                    tgt2src=tgt2src[:3, :],
                    src_K=scene['fxfycxcy'][idx1],
                    tgt_K=scene['fxfycxcy'][idx2],
                    imsize=64
                )

                src_images.append(src_img_resized)
                tgt_images.append(tgt_img_resized)
                geometries.append(geo)

            except Exception:
                continue

        if not src_images: return None

        return {
            'src_image': torch.stack(src_images),
            'tgt_image': torch.stack(tgt_images),
            'geometry': torch.stack(geometries),
        }




class DualSourceCollate:
   
    USE_INTERPOLATION_MODE = False

    def __call__(self, batch: list[dict]) -> dict | None:
        src_images, tgt_images, geometries = [], [], []

        if self.USE_INTERPOLATION_MODE:
            
            valid_scenes = [s for s in batch if s and 'image' in s and s['image'].shape[0] >= 8]
            if not valid_scenes: return None

            for scene in valid_scenes:
                try:
                    num_available = scene['image'].shape[0]
                    # NOTE: This logic uses 6 targets per scene by default.
                    min_frame_dist, max_frame_dist, num_targets = 25, min(num_available - 1, 100), 6

                    if max_frame_dist <= min_frame_dist: continue

                    frame_dist = random.randint(min_frame_dist, max_frame_dist)
                    src1_idx = random.randint(0, num_available - frame_dist - 1)
                    src2_idx = src1_idx + frame_dist

                    if (src2_idx - src1_idx - 1) < num_targets: continue

                    target_indices = random.sample(range(src1_idx + 1, src2_idx), num_targets)
                    src1_img_resized = F.resize(scene['image'][src1_idx], [64, 64], antialias=True) * 255.0
                    src2_img_resized = F.resize(scene['image'][src2_idx], [64, 64], antialias=True) * 255.0
                    src1_c2w, src2_c2w = torch.as_tensor(scene['c2w'][src1_idx]), torch.as_tensor(scene['c2w'][src2_idx])
                    src1_K, src2_K = scene['fxfycxcy'][src1_idx], scene['fxfycxcy'][src2_idx]

                    for tgt_idx in target_indices:
                        tgt_img_resized = F.resize(scene['image'][tgt_idx], [64, 64], antialias=True).float()
                        tgt_c2w, tgt_K = torch.as_tensor(scene['c2w'][tgt_idx]), scene['fxfycxcy'][tgt_idx]

                        # Pair 1: Source 1 -> Target
                        tgt2src1 = torch.linalg.inv(tgt_c2w) @ src1_c2w
                        geo1 = compose_geometry(tgt2src=tgt2src1[:3, :], src_K=src1_K, tgt_K=tgt_K, imsize=64)
                        src_images.append(src1_img_resized)
                        tgt_images.append(tgt_img_resized)
                        geometries.append(geo1)

                        # Pair 2: Source 2 -> Target
                        tgt2src2 = torch.linalg.inv(tgt_c2w) @ src2_c2w
                        geo2 = compose_geometry(tgt2src=tgt2src2[:3, :], src_K=src2_K, tgt_K=tgt_K, imsize=64)
                        src_images.append(src2_img_resized)
                        tgt_images.append(tgt_img_resized)
                        geometries.append(geo2)

                except Exception:
                    continue
        else:
           
            valid_scenes = [s for s in batch if s and 'image' in s and s['image'].shape[0] >= 3]
            if not valid_scenes: return None

            for scene in valid_scenes:
                try:
                    num_available = scene['image'].shape[0]
                    
                    src1_idx, src2_idx, tgt_idx = random.sample(range(num_available), 3)

                    # Load images, resize, and FIX: convert to float32
                    src1_img_resized = F.resize(scene['image'][src1_idx], [64, 64], antialias=True) * 255.0
                    src2_img_resized = F.resize(scene['image'][src2_idx], [64, 64], antialias=True) * 255.0
                    tgt_img_resized = F.resize(scene['image'][tgt_idx], [64, 64], antialias=True) * 255.0

                    # Load camera parameters
                    src1_c2w, src2_c2w, tgt_c2w = (
                        torch.as_tensor(scene['c2w'][idx]) for idx in [src1_idx, src2_idx, tgt_idx]
                    )
                    src1_K, src2_K, tgt_K = (
                        scene['fxfycxcy'][idx] for idx in [src1_idx, src2_idx, tgt_idx]
                    )

                    # Pair 1: Source 1 -> Target
                    tgt2src1 = torch.linalg.inv(tgt_c2w) @ src1_c2w
                    geo1 = compose_geometry(tgt2src=tgt2src1[:3, :], src_K=src1_K, tgt_K=tgt_K, imsize=64)
                    src_images.append(src1_img_resized)
                    tgt_images.append(tgt_img_resized)
                    geometries.append(geo1)

                    # Pair 2: Source 2 -> Target
                    tgt2src2 = torch.linalg.inv(tgt_c2w) @ src2_c2w
                    geo2 = compose_geometry(tgt2src=tgt2src2[:3, :], src_K=src2_K, tgt_K=tgt_K, imsize=64)
                    src_images.append(src2_img_resized)
                    tgt_images.append(tgt_img_resized) # Use the same target image again
                    geometries.append(geo2)

                except Exception:
                    continue

        if not src_images: return None

        final_batch_dict = {
            'src_image': torch.stack(src_images),
            'tgt_image': torch.stack(tgt_images),
            'geometry': torch.stack(geometries),
        }
        
        print("\n--- [LitData Collate Output] ---")
        img_tensor = final_batch_dict['src_image']
        print(f"dtype={img_tensor.dtype}, shape={img_tensor.shape}, min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
        print("--------------------------------\n")

        return final_batch_dict


class CustomLitDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, cache_dir, max_cache_size="160GB", **kwargs):
        super().__init__()
        # shuffle=True is now safe and recommended for both modes.
        self.streaming_dataset = ld.StreamingDataset(
            input_dir=path, cache_dir=cache_dir, max_cache_size=max_cache_size, shuffle=True
        )

    def __iter__(self):
        while True:
            yield from self.streaming_dataset