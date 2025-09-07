import torch
import litdata as ld
import random
from training.utils import compose_geometry
import torchvision.transforms.functional as F

# --- GLOBAL TOGGLE ---
VANILLA_MODE = False


def process_scene_vanilla(scene: dict):
    try: 
        if not scene or 'image' not in scene or scene['image'].shape[0] < 2:
            return None

        num_available = scene['image'].shape[0]
        idx1, idx2 = random.sample(range(num_available), 2)

        src_img_resized = F.resize(scene['image'][idx1], [64, 64], antialias=True)
        tgt_img_resized = F.resize(scene['image'][idx2], [64, 64], antialias=True)
        
        src_c2w = torch.as_tensor(scene['c2w'][idx1], dtype=torch.float32)
        tgt_c2w = torch.as_tensor(scene['c2w'][idx2], dtype=torch.float32)
        
        tgt2src = torch.linalg.inv(tgt_c2w) @ src_c2w
        
        geo = compose_geometry(
            tgt2src=tgt2src[:3, :],
            src_K=scene['fxfycxcy'][idx1],
            tgt_K=scene['fxfycxcy'][idx2],
            imsize=64
        )
        
        processed_item = {
            'src_image': src_img_resized,
            'tgt_image': tgt_img_resized,
            'geometry': geo
        }
        
        if 'sr_image' in scene:
            processed_item['high_res_src_image'] = scene['sr_image'][idx1]
            
        return processed_item

    except torch.linalg.LinAlgError:
        return None
    except Exception as e:
        return None



def process_scene_interpolation_stacked(scene: dict, num_targets: int = 6):
    
    try:
        if not scene or 'image' not in scene or scene['image'].shape[0] < (num_targets + 2):
            return

        num_available = scene['image'].shape[0]

        min_frame_dist = 25
        max_frame_dist = min(num_available - 1, 100)
        if max_frame_dist <= min_frame_dist: return

        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        src1_idx = random.randint(0, num_available - frame_dist - 1)
        src2_idx = src1_idx + frame_dist
        
        num_intermediate_views = src2_idx - src1_idx - 1
        if num_intermediate_views < num_targets: return
        
        target_indices = random.sample(range(src1_idx + 1, src2_idx), num_targets)

        # Pre-process sources once per scene.
        src1_img_resized = F.resize(scene['image'][src1_idx], [64, 64], antialias=True)
        src2_img_resized = F.resize(scene['image'][src2_idx], [64, 64], antialias=True)
        src1_c2w = torch.as_tensor(scene['c2w'][src1_idx], dtype=torch.float32)
        src2_c2w = torch.as_tensor(scene['c2w'][src2_idx], dtype=torch.float32)
        src1_K = scene['fxfycxcy'][src1_idx]
        src2_K = scene['fxfycxcy'][src2_idx]
        high_res_src1 = scene.get('sr_image', scene['image'])[src1_idx]
        high_res_src2 = scene.get('sr_image', scene['image'])[src2_idx]
        
        for tgt_idx in target_indices:
            tgt_img_resized = F.resize(scene['image'][tgt_idx], [64, 64], antialias=True)
            tgt_c2w = torch.as_tensor(scene['c2w'][tgt_idx], dtype=torch.float32)
            tgt_K = scene['fxfycxcy'][tgt_idx]

            # Calculate two sets of relative geometries.
            tgt2src1 = torch.linalg.inv(tgt_c2w) @ src1_c2w
            tgt2src2 = torch.linalg.inv(tgt_c2w) @ src2_c2w
            geo1 = compose_geometry(tgt2src=tgt2src1[:3, :], src_K=src1_K, tgt_K=tgt_K, imsize=64)
            geo2 = compose_geometry(tgt2src=tgt2src2[:3, :], src_K=src2_K, tgt_K=tgt_K, imsize=64)

            # Yield the first item for the pair (target + source 1)
            yield {
                'src_image': src1_img_resized,
                'tgt_image': tgt_img_resized,
                'geometry': geo1,
                'high_res_src_image': high_res_src1,
            }
            # Yield the second item for the pair (target + source 2)
            yield {
                'src_image': src2_img_resized,
                'tgt_image': tgt_img_resized,
                'geometry': geo2,
                'high_res_src_image': high_res_src2,
            }

    except Exception:
        return




class CustomLitDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, cache_dir, max_cache_size="160GB", mode='vanilla', **kwargs):
        super().__init__()
        self.streaming_dataset = ld.StreamingDataset(
            input_dir=path, cache_dir=cache_dir, max_cache_size=max_cache_size, shuffle=False
        )
        self.mode = mode # <-- Store the mode passed during creation
        print(f"LitData StreamingDataset initialized in {self.mode.upper()} MODE.")

    def __iter__(self):
        while True:
            for raw_item in self.streaming_dataset:
                if self.mode == 'vanilla':
                    processed_item = process_scene_vanilla(raw_item)
                    if processed_item is not None: yield processed_item
                else:
                    yield from process_scene_interpolation_stacked(raw_item, num_targets=6)




class CustomLitCollate:
    def __call__(self, batch: list[dict]) -> dict | None:
        batch = [item for item in batch if item is not None]
        if not batch: return None
        
        final_batch = {
            'src_image': torch.stack([s['src_image'] for s in batch]),
            'tgt_image': torch.stack([s['tgt_image'] for s in batch]),
            'geometry': torch.stack([s['geometry'] for s in batch]),
        }
        if 'high_res_src_image' in batch[0]:
            final_batch['high_res_src_image'] = torch.stack([s['high_res_src_image'] for s in batch])
        return final_batch