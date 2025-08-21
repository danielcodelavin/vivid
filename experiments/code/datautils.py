#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024-2025 Apple Inc. All Rights Reserved.
#

from glob import glob
import os
import random
import PIL
import numpy as np
import torch
from torch.utils import data
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms.functional
from kornia.geometry.conversions import quaternion_from_euler, quaternion_to_rotation_matrix
from kornia.geometry.transform import warp_affine, warp_perspective

from training.utils import compose_geometry, expand_extrinsics


def pil_to_tensor(img):
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()


def parse_line(line, width=640, height=360):
    items = line.split(" ")
    timestamp = items[0]
    focal_length_x, focal_length_y, principal_point_x, principal_point_y = map(float, items[1:5])
    K = torch.tensor([
                      [width * focal_length_x, 0, width * principal_point_x],
                      [0, height * focal_length_y, height * principal_point_y],
                      [0, 0, 1],
                     ])
    camera_pose = torch.tensor(list(map(float, items[7:]))).reshape(3, 4)
    return timestamp, K, camera_pose


def transform_coordinates(K, camera_pose, new_size, center_crop_size, old_width=640, old_height=360):
    # crop operation
    new_corner = torch.tensor([(old_width - center_crop_size) // 2, (old_height - center_crop_size) // 2])
    K[:2, 2] = K[:2, 2] - new_corner
    # resize operation
    K[:2] = new_size * K[:2] / center_crop_size 
    return K, camera_pose


def generate_rotation_matrix(generator, max_pitch: float, max_yaw: float, max_roll: float):
    # create the rotation matrix
    roll = torch.rand((1,), generator=generator, dtype=torch.float32) * 2 * max_roll - max_roll
    pitch = torch.rand((1,), generator=generator, dtype=torch.float32) * 2 * max_pitch - max_pitch
    yaw = torch.rand((1,), generator=generator, dtype=torch.float32) * 2 * max_yaw - max_yaw
    # duplicate the same angle for all the frames in the sequence
    quat_coeffs = quaternion_from_euler(pitch, yaw, roll)
    quat = torch.cat(quat_coeffs, dim=-1)
    R = quaternion_to_rotation_matrix(quat)
    return R


def calc_homography_for_rotation(R, K):
    return K @ R @ torch.linalg.inv(K)


def random_camera_rotation(image, extrinsics, intrinsics, generator, max_angle_pitch=0, max_angle_yaw=10, max_angle_roll=0):
    # create the rotation matrix
    R = generate_rotation_matrix(generator, max_angle_pitch / 180 * torch.pi, max_angle_yaw / 180 * torch.pi, max_angle_roll / 180 * torch.pi)
    # calculate the resulting homography matrix
    H = calc_homography_for_rotation(R, intrinsics)
    # apply homography on images
    rotated_image = warp_perspective(image[None], H[None], (image.shape[-2], image.shape[-1]), mode="bilinear")[0]
    # apply homography on extrinsics
    rotated_extrinsics = torch.cat((R @ extrinsics[:, :3], R @ extrinsics[:, 3:]), dim=1)
    # return new image and extrinsics
    return rotated_image, rotated_extrinsics


def transforms(im, center_crop_size, imsize):
    return torchvision.transforms.functional.resize(torchvision.transforms.functional.center_crop(im, center_crop_size), imsize)


def nvs_transforms(src_image, src_intrinsics, src_extrinsics, tgt_image, tgt_intrinsics, tgt_extrinsics, center_crop_size, imsize, srsize):
    old_height, old_width = src_image.shape[-2:]
    center_crop_size = min(old_height, old_width)
    sr_src_intrinsics, sr_src_extrinsics = transform_coordinates(src_intrinsics, src_extrinsics, new_size=srsize, center_crop_size=center_crop_size, old_width=old_width, old_height=old_height)
    sr_tgt_intrinsics, sr_tgt_extrinsics = transform_coordinates(tgt_intrinsics, tgt_extrinsics, new_size=srsize, center_crop_size=center_crop_size, old_width=old_width, old_height=old_height)
    tgt_intrinsics, tgt_extrinsics = transform_coordinates(tgt_intrinsics, tgt_extrinsics, new_size=imsize, center_crop_size=center_crop_size, old_width=old_width, old_height=old_height)
    src_intrinsics, src_extrinsics = transform_coordinates(src_intrinsics, src_extrinsics, new_size=imsize, center_crop_size=center_crop_size, old_width=old_width, old_height=old_height)
    sr_src_image = transforms(src_image, center_crop_size=center_crop_size, imsize=srsize)
    sr_tgt_image = transforms(tgt_image, center_crop_size=center_crop_size, imsize=srsize)
    src_image = transforms(src_image, center_crop_size=center_crop_size, imsize=imsize)
    tgt_image = transforms(tgt_image, center_crop_size=center_crop_size, imsize=imsize)

    normalized_extrinsics = (expand_extrinsics(src_extrinsics) @ torch.inverse(expand_extrinsics(tgt_extrinsics)))[:3]
    sr_normalized_extrinsics = (expand_extrinsics(sr_src_extrinsics) @ torch.inverse(expand_extrinsics(sr_tgt_extrinsics)))[:3]
    geometry = compose_geometry(normalized_extrinsics, src_intrinsics, tgt_intrinsics, imsize=imsize)
    sr_geometry = compose_geometry(sr_normalized_extrinsics, sr_src_intrinsics, sr_tgt_intrinsics, imsize=srsize)

    output = {"src_image": src_image, "tgt_image": tgt_image, "geometry": geometry, "sr_src_image": sr_src_image, "sr_tgt_image": sr_tgt_image, "sr_geometry": sr_geometry}
    return output


class RealEstate10K(data.Dataset):
    def __init__(self, split="train", imsize=64, data_root="data", sr_mult=4, range_selection=None, **kwargs):
        self.split = split
        self.imsize = imsize
        self.srsize = imsize * sr_mult
        self.num_channels = 3
        self.data_root = data_root
        self.sequence_dir = os.path.join(self.data_root, "RealEstate10K", self.split)
        self.sequence_paths = glob(os.path.join(self.sequence_dir, "*.txt"))
        self.sequence_paths = [path for path in self.sequence_paths if os.path.exists(os.path.join(self.data_root, self.split, os.path.basename(path).replace(".txt", "")))]
        self.range_selection = range_selection
        if self.range_selection is not None:
            s, e = {"mid": (30, 60), "long": (60, 120)}[self.range_selection]
            self.sequence_paths = [path for path in self.sequence_paths if len(open(path, "r").read().splitlines()) > (s + 1)]

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, index):
        seq_path = self.sequence_paths[index]
        seq = os.path.basename(seq_path).replace(".txt", "")
        with open(seq_path, "r") as thefile:
            lines = thefile.read().splitlines()
        lines.pop(0)
        weight = torch.tensor([1.0] * len(lines))
        if self.range_selection is None:
            indexes = torch.multinomial(weight, num_samples=2, replacement=(len(lines) == 1))
        else:
            s, e = {"mid": (30, 60), "long": (60, 120)}[self.range_selection]
            weight = torch.zeros((len(lines),))
            weight[:len(weight)-s] = weight[s:] = 1
            index1 = torch.cat([torch.multinomial(weight, num_samples=1)])
            weight = torch.zeros((len(lines),))
            weight[index1+s:index1+e] = weight[max(0, index1-e):max(0,1 + index1-s)] = 1
            indexes = torch.cat([index1, torch.multinomial(weight, num_samples=1)])

        src_timestamp, src_intrinsics, src_extrinsics = parse_line(lines[indexes[0].item()])
        tgt_timestamp, tgt_intrinsics, tgt_extrinsics = parse_line(lines[indexes[1].item()])
        src_image = pil_to_tensor(PIL.Image.open(os.path.join(self.data_root, self.split, seq, src_timestamp + ".png")))
        tgt_image = pil_to_tensor(PIL.Image.open(os.path.join(self.data_root, self.split, seq, tgt_timestamp + ".png")))

        return nvs_transforms(src_image, src_intrinsics, src_extrinsics, tgt_image, tgt_intrinsics, tgt_extrinsics, 360, self.imsize, self.srsize)


class SingleImages(data.Dataset):
    def __init__(self, imsize, data_root="data", sr_mult=4, **kwargs):
        self.imsize = imsize
        self.srsize = imsize * sr_mult
        self.data_root = data_root
        self.image_dir = os.path.join(self.data_root, "SingleImages")
        self.image_path = glob(os.path.join(self.sequence_dir, "*.png")) + glob(os.path.join(self.sequence_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        src_image = pil_to_tensor(PIL.Image.open(self.image_path[index]))
        focal_length_x, focal_length_y = 0.6, 0.6
        src_intrinsics = torch.tensor([
                        [width * focal_length_x, 0, width * 0.5],
                        [0, height * focal_length_y, height * 0.5],
                        [0, 0, 1.],
                        ])
        src_extrinsics = torch.tensor([
                        [1., 0, 0, 0],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0],
                        ])

        dst_intrinsics = src_intrinsics.clone()
        if torch.rand((1,)) < 0.5:    
            center_crop_size = 320
            dst_image, dst_extrinsics = random_camera_rotation(src_image, src_extrinsics, src_intrinsics, max_angle_pitch=8.3, max_angle_yaw=8.3, max_angle_roll=3.5)
            src_image, src_extrinsics = random_camera_rotation(src_image, src_extrinsics, src_intrinsics, max_angle_pitch=8.3, max_angle_yaw=8.3, max_angle_roll=3.5)
        else:
            center_crop_size = 384
            dst_image, dst_extrinsics = random_camera_rotation(src_image, src_extrinsics, src_intrinsics, max_angle_pitch=5.5, max_angle_yaw=5.5, max_angle_roll=0)
            src_image, src_extrinsics = random_camera_rotation(src_image, src_extrinsics, src_intrinsics, max_angle_pitch=5.5, max_angle_yaw=5.5, max_angle_roll=0)

        return nvs_transforms(src_image, src_intrinsics, src_extrinsics, tgt_image, tgt_intrinsics, tgt_extrinsics, center_crop_size, self.imsize, self.srsize)


class ImageFolderDataset(data.Dataset):
    def __init__(self, dir, max_size=None, random_seed=0):
        self.image_paths = sorted(glob(os.path.join(self.sequence_dir, "sample_*.png")) + glob(os.path.join(self.sequence_dir, "sample_*.jpg")), key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group()))
        if max_size is not None:
            random.seed(random_seed)  # Set the seed for reproducibility
            self.image_paths = sorted(random.sample(self.image_paths, max_size), key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group()))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        return (pil_to_tensor(PIL.Image.open(self.image_path[index].replace("sample", prefix))) for prefix in ["src", "tgt", "sample"])
