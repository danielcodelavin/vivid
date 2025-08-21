# In file: training/datautils/custom_litdata_loader.py

import torch
import litdata as ld
import random
from training.utils import compose_geometry

# ======================================================================================
# Configuration Flag
# ======================================================================================
VANILLA_MODE = True
NUM_INPUT_VIEWS = 2
NUM_TARGET_VIEWS_TOTAL = 6
# ======================================================================================

class CustomLitCollate:
    def __call__(self, batch):
        valid_samples = [s for s in batch if s and 'image' in s and s['image'].shape[0] >= 2]
        if not valid_samples:
            return None

        final_batch = {}

        if VANILLA_MODE:
            src_images, tgt_images, geometries, high_res_src = [], [], [], []

            for scene in valid_samples:
                num_available = scene['image'].shape[0]
                idx1, idx2 = random.sample(range(num_available), 2)
                
                src_images.append(scene['image'][idx1])
                tgt_images.append(scene['image'][idx2])
                
                
                src_c2w = torch.as_tensor(scene['c2w'][idx1], dtype=torch.float32)
                tgt_c2w = torch.as_tensor(scene['c2w'][idx2], dtype=torch.float32)

                
                tgt2src = torch.linalg.inv(tgt_c2w) @ src_c2w

                
                geo = compose_geometry(
                    tgt2src=tgt2src[:3, :], # Pass the 3x4 relative pose matrix
                    src_K=scene['fxfycxcy'][idx1],
                    tgt_K=scene['fxfycxcy'][idx2],
                    imsize=64  
                )
                
                
                geometries.append(geo)
                
                if 'sr_image' in scene:
                    high_res_src.append(scene['sr_image'][idx1])

            final_batch['src_image'] = torch.stack(src_images)
            final_batch['tgt_image'] = torch.stack(tgt_images)
            final_batch['geometry'] = torch.stack(geometries)
            if high_res_src:
                final_batch['high_res_src_image'] = torch.stack(high_res_src)

        else:
           
            return None

        return final_batch


class CustomLitDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_dir, max_cache_size="50GB", **kwargs):
        super().__init__()
        print(f"Initializing LitData StreamingDataset from path: {path}")
        self.dataset = ld.StreamingDataset(
            input_dir=path,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            shuffle=True,
        )
        self._len = len(self.dataset)
        print(f"Found {self._len} items in dataset.")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.dataset[idx]