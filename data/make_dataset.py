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

# %load_ext autoreload
# %autoreload 2

# +
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation
from tqdm.auto import tqdm
# -

from .tools import CS_Dataset, Image_path_reader

# # Settings

# ## path for each dataset

DATA_PATH = {
    "coco": "/usr/local/ay_data/dataset/1-CS/COCO",
    "set5": "/usr/local/ay_data/dataset/1-CS/Set5",
    "set11": "/usr/local/ay_data/dataset/1-CS/Set11",
    "set14": "/usr/local/ay_data/dataset/1-CS/Set14",
    "bsd68": "/usr/local/ay_data/dataset/1-CS/BSD68",
    "bsd100": "/usr/local/ay_data/dataset/1-CS/BSD100",
    "cbsd68": "/usr/local/ay_data/dataset/1-CS/CBSD68",
    "train400": "/usr/local/ay_data/dataset/1-CS/train400",
    "urban100": "/usr/local/ay_data/dataset/1-CS/Urban100",
    'set1024': "/usr/local/ay_data/dataset/1-CS/Set1024",
    'test_for_visual_comparison':"/usr/local/ay_data/dataset/1-CS/test_for_visual_comparison",
}


# # Data Augmentation 

def get_transfroms(cfg_aug):
    if cfg_aug is None:
        print("No augmentation is used.")
        return None
    aug = []
    if cfg_aug.flip:
        aug.append(RandomHorizontalFlip(p=0.5))
    if cfg_aug.rotate:
        aug.append(RandomRotation(degrees=30))
    print(
        "data augmentation: ",
        ", ".join([key for key in cfg_aug.keys() if cfg_aug[key]]),
    )
    return Compose(aug)


# # Create dataloader

# ## test & val

def get_testset(dataset_name=None, data=None):
    if data is None:
        assert dataset_name is not None
        root_path = DATA_PATH[dataset_name]
        data = Image_path_reader.read_paths(root_path)
    dataset = CS_Dataset(data, img_size=-1, transforms=None)
    dl = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=4
    )
    return dl


# ## train

# +
import random

from torch.utils.data.sampler import Sampler


class ImageSampler(Sampler):
    def __init__(self, train_num=40000, n_samples=40000):

        self.train_num = train_num
        self.n_samples = n_samples

        
        self.indices = [x % train_num for x in list(range(n_samples))]

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return self.n_samples


# -

def get_trainset(cfg_ds, cfg_aug=None):
    root_path = DATA_PATH[cfg_ds.trainset]
    data = Image_path_reader.read_paths(root_path, n_img=cfg_ds.train_num)

    train_data = data
    val_data = None

    if cfg_ds.val_rate > 0:
        val_num = (
            int(cfg_ds.train_num * cfg_ds.val_rate)
            if cfg_ds.val_rate < 1
            else cfg_ds.val_rate
        )
        train_data = data.query(f"id < {cfg_ds.train_num - val_num}").reset_index(
            drop=True
        )
        val_data = data.query(f"id >= {cfg_ds.train_num - val_num}").reset_index(
            drop=True
        )

    res = {}
    transforms = get_transfroms(cfg_aug=cfg_aug)
    dataset = CS_Dataset(
        train_data,
        img_size=cfg_ds.train_size,
        transforms=transforms,
    )
    if cfg_ds.n_samples_per_epoch == len(train_data):
        sampler = None
    else:
        sampler = ImageSampler(
            train_num=len(train_data), n_samples=cfg_ds.n_samples_per_epoch
        )

    res["train"] = DataLoader(
        dataset,
        batch_size=cfg_ds.batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    if val_data is not None:
        res["val"] = get_testset(data=val_data)
    return res
