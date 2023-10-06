# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ToTensor,
)
from tqdm.auto import tqdm


# -

# # Methods

# ## Rgb2YCbCr

def rgb2ycbcr(img, only_y=True):
    """计算YCbCr，和 ``matlab rgb2ycbcr`` 的运算结果一样

    Args:
        img(uint8, float): the input image. (H, W, C)
        only_y: only return Y channel

    Returns:
        the converted image: (H, W) if only_y, else (H, W, C)
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


# ## Read image from path

def read_img(img_path: str, mode: str = "L") -> np.ndarray:
    """读取本地图像

    Args:
        img_path: the local path of the image
        mode: ['L', 'YCbCr_Y']

    Returns:
        img: (C, H, W)

    """
    assert mode in ["L", "YCbCr_Y"]

    img_org = Image.open(img_path)

    if len(img_org.getbands()) == 1:
        img_org = np.array(img_org)
        return img_org[:, :, None]
    else:
        if mode == "YCbCr_Y":
            img_org = np.array(img_org)
            return rgb2ycbcr(img_org)[:, :, None]
        elif mode == "L":
            img = img_org.convert(mode)
            img = np.array(img)
            return img[:, :, None]
        else:
            img_org = np.array(img_org)
            return img_org


# ## Read all img paths from folders

class Image_path_reader:
    exts = [".jpg", ".png", ".tif", ".tiff"]

    @staticmethod
    def read_paths(root_paths, n_img=-1):
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        data = pd.concat(
            [Image_path_reader._read_img_paths(path) for path in root_paths]
        )
        data = data.reset_index(drop=True)
        data = Image_path_reader.assign_id(data)
        if n_img > 0:
            data = data.query(f"id < {n_img}").reset_index(drop=True)
        return data

    @staticmethod
    def _read_img_paths(root_path):
        """
        read all img paths in the root_path
        """
        img_names = sorted(os.listdir(root_path))
        img_names = [
            name
            for name in img_names
            if os.path.splitext(name)[1] in Image_path_reader.exts
        ]
        data = pd.DataFrame()
        data["name"] = img_names
        data["path"] = data["name"].apply(lambda x: os.path.join(root_path, x))
        return data

    @staticmethod
    def assign_id(data):
        indices = np.arange(len(data))
        np.random.seed(42)
        np.random.shuffle(indices)
        data["id"] = indices
        return data


# + tags=["active-ipynb", "style-commentate"]
# Image_path_reader.read_paths("/usr/local/ay_data/dataset/1-CS/COCO", 100)
# -

# # torch Dataset for image

class CS_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, img_size=-1, transforms=None):
        """
        Args:
            root_path: the root paths for the img for training
            n_img: the number of used images for training.
            transform: the image transforms after reading the image.

        """
        super().__init__()
        self.data = data

        self.to_tensor = ToTensor()  # Convert the image to PyTorch tensor
        self.transforms = transforms
        self.random_crop = (
            RandomCrop((img_size, img_size), pad_if_needed=True)
            if img_size >= 0
            else None
        )
        self.img_size = img_size
        
    def set_img_size(self, img_size):
        print(f"Change img_size from {self.img_size} to {img_size}!!!!")
        self.random_crop = (
            RandomCrop((img_size, img_size), pad_if_needed=True)
            if img_size >= 0
            else None
        )
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < self.__len__()

        item = self.data.iloc[idx]
        name = item["name"]
        img_path = item["path"]

        img = read_img(img_path, mode="YCbCr_Y")
        # print(img.shape)

        img = self.to_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.random_crop is not None:
            img = self.random_crop(img)

        return img, name
