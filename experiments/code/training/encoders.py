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


import os
import warnings
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')


@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

