# In file: training/datautils/custom_litdata_loader.py

import torch
import litdata as ld
import random
from training.utils import compose_geometry
import torchvision.transforms.functional as F

VANILLA_MODE = True

def process_scene_vanilla(scene: dict) -> dict | None:
    """
    This function handles all CPU-heavy processing for a SINGLE data sample.
    It is called from __getitem__, which is executed in parallel by the DataLoader workers.
    """
    try: 
        if not scene or 'image' not in scene or scene['image'].shape[0] < 2:
            return None

        num_available = scene['image'].shape[0]
        idx1, idx2 = random.sample(range(num_available), 2)

        src_img_resized = F.resize(scene['image'][idx1], [64, 64], antialias=True)
        tgt_img_resized = F.resize(scene['image'][idx2], [64, 64], antialias=True)
        
        src_c2w = torch.as_tensor(scene['c2w'][idx1], dtype=torch.float32)
        tgt_c2w = torch.as_tensor(scene['c2w'][idx2], dtype=torch.float32)
        
        # This is the line that can fail
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


class CustomLitDataset(torch.utils.data.IterableDataset):
    """
    An efficient IterableDataset for a single, large LitData chunk.
    It relies on LitData's internal shuffling and parallel processing.
    """
    def __init__(self, path, cache_dir, max_cache_size="160GB", **kwargs):
        super().__init__()
        
        # LitData's shuffle=True is designed to be efficient by shuffling a buffer of chunks internally.
        # This is the correct way to handle a single, monolithic dataset chunk.
        self.streaming_dataset = ld.StreamingDataset(
            input_dir=path,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            shuffle=True, 
        )
        print(f"LitData StreamingDataset initialized from a single chunk at {path}.")
        print(f"Internal shuffling is enabled. Cache size is {max_cache_size}.")

    def __iter__(self):
        """Yields fully processed items, ready for the collate function."""
        for raw_item in self.streaming_dataset:
            if VANILLA_MODE:
                processed_item = process_scene_vanilla(raw_item)
                if processed_item is not None:
                    yield processed_item
            else:
                # TODO: Implement non-vanilla logic
                pass


class CustomLitCollate:
    """A lean collate function that only stacks already-processed items."""
    def __call__(self, batch: list[dict]) -> dict | None:
       
        batch = [item for item in batch if item is not None]
        
        if not batch:
            return None # Return None if the entire batch was filtered out
    
        
        if VANILLA_MODE:
            final_batch = {
                'src_image': torch.stack([s['src_image'] for s in batch]),
                'tgt_image': torch.stack([s['tgt_image'] for s in batch]),
                'geometry': torch.stack([s['geometry'] for s in batch]),
            }
            if 'high_res_src_image' in batch[0]:
                final_batch['high_res_src_image'] = torch.stack([s['high_res_src_image'] for s in batch])
            return final_batch
        else:
            return None