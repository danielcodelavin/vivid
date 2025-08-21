# In file: training/datautils/custom_litdata_loader.py

import torch
import litdata as ld
import random
from training.utils import compose_geometry


VANILLA_MODE = True


NUM_INPUT_VIEWS = 2
NUM_TARGET_VIEWS_TOTAL = 6


class CustomLitCollate:
    """
    This class processes a batch of scenes from the CustomLitDataset.
    It selects views based on the VANILLA_MODE flag and formats them
    for the VIVID training loop.
    """
    

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
                
                geo = compose_geometry(
                    scene['c2w'][idx2], scene['fxfycxcy'][idx2],
                    scene['c2w'][idx1], scene['fxfycxcy'][idx1]
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
            # --- LVSM Multi-view Loading Logic (2 inputs, 1 of 6 targets per step) ---
            src1_list, src2_list, tgt_list, geo1_list, geo2_list = [], [], [], [], []
            # high_res1_list, high_res2_list = [], [] # For depth model

            min_required = NUM_INPUT_VIEWS + NUM_TARGET_VIEWS_TOTAL
            for scene in valid_samples:
                num_available = scene['image'].shape[0]
                if num_available < min_required:
                    continue

                all_indices = list(range(num_available))
                random.shuffle(all_indices)
                
                # Assign indices for sources and the pool of potential targets
                src_indices = all_indices[:NUM_INPUT_VIEWS]
                target_pool_indices = all_indices[NUM_INPUT_VIEWS:min_required]

                
                chosen_tgt_idx = random.choice(target_pool_indices)
                
                src1_idx, src2_idx = src_indices[0], src_indices[1]

                # Append images to lists
                src1_list.append(scene['image'][src1_idx])
                src2_list.append(scene['image'][src2_idx])
                tgt_list.append(scene['image'][chosen_tgt_idx])
                
                # Compose geometries relative to the chosen target
                geo1 = compose_geometry(
                    scene['c2w'][chosen_tgt_idx], scene['fxfycxcy'][chosen_tgt_idx],
                    scene['c2w'][src1_idx], scene['fxfycxcy'][src1_idx]
                )
                geo2 = compose_geometry(
                    scene['c2w'][chosen_tgt_idx], scene['fxfycxcy'][chosen_tgt_idx],
                    scene['c2w'][src2_idx], scene['fxfycxcy'][src2_idx]
                )
                geo1_list.append(geo1)
                geo2_list.append(geo2)


            if not src1_list: return None

            final_batch['src_image_1'] = torch.stack(src1_list)
            final_batch['src_image_2'] = torch.stack(src2_list)
            final_batch['tgt_image'] = torch.stack(tgt_list)
            final_batch['geometry_1'] = torch.stack(geo1_list)
            final_batch['geometry_2'] = torch.stack(geo2_list)
            # if high_res1_list:
            #     final_batch['high_res_1'] = torch.stack(high_res1_list)
            #     final_batch['high_res_2'] = torch.stack(high_res2_list)

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