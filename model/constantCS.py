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
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
# -

import pytorch_lightning as pl
from .psnr_ssim import PSNR_SSIM



def validate_path(path):
    """检查文件或者文件夹的路径是否有效
        
        如果路径不存在，将会创建它
        
        Args:
            path(str): the path of folder
            
    """
    folder = os.path.split(path)[0]
    if not os.path.isdir(folder):
        os.makedirs(folder)

class ConstantCS(pl.LightningModule):
    def __init__(self, model, sr, blk_size=32, args=None):
        super().__init__()

        self.sr = sr
        self.blk_size = blk_size
        self.model = model

        
        prefix = ["train", "val", "test"]
        self.train_metrics, self.val_metrics, self.test_metrics = [
            PSNR_SSIM(prefix=prefix[i]) for i in range(3)
        ]

    def configure_optimizers(self):
        raise NotImplementedError

    def get_loss(self, model_res, x_org):
        raise NotImplementedError

    def pad_img(self, x):
        B, C, H, W = x.shape

        # calculate new height and width
        new_height = H + (self.blk_size - H % self.blk_size) % self.blk_size
        new_width = W + (self.blk_size - W % self.blk_size) % self.blk_size

        # Pad image to new height and width
        padded_image = nn.functional.pad(x, (0, new_width - W, 0, new_height - H))
        return padded_image

    def _shared_eval_step(self, batch, batch_idx, metrics=None, stage="train"):
        x_org, img_name = batch  # get image x from the input batch

        used_time = 0
        if stage == "test":
            start_time = time.time()

        # first padd image if needed (in validation and test).
        # then sample and reconstruct image
        B, C, H, W = x_org.shape
        if stage != "train":
            x_org = self.pad_img(x_org)
        model_res = self.model(x_org)

        
        if stage == "test":
            end_time = time.time()
            used_time = end_time - start_time

        # calculate loss, psnr, ssim, and log them in the logger file.
        res = {}
        res[stage + "-loss"] = self.get_loss(model_res, x_org)
        if stage != "train":
            model_res["x_re"] = model_res["x_re"][:, :, 0:H, 0:W]
            x_org = x_org[:, :, 0:H, 0:W]

        return self.log_info(
            res, x_org, model_res["x_re"], metrics, img_name, stage, used_time=used_time
        )

    def log_info(self, res, x_org, x_re, metrics, img_name, stage, **kwargs):
        if metrics is not None:
            x_re = (x_re * 255).to(torch.uint8).to(torch.float32)
            x_org = (x_org * 255).to(torch.uint8).to(torch.float32)
            _metrics = metrics.update(x_org, x_re)

        if stage == "test":
            if hasattr(self, "save_test_image") and self.save_test_image:
                img_name = os.path.splitext(img_name[0])[0]
                file_name = "%d-%s-%.2f-%.4f.png" % (
                    self.sr * 100,
                    self.model_name,
                    _metrics["psnr"].item(),
                    _metrics["ssim"].item(),
                )
                save_path = os.path.join(self.save_dir, img_name, file_name)
                validate_path(save_path)
                from skimage.io import imsave
                imsave(save_path, x_re[0, 0].detach().cpu().numpy().astype(np.uint8))

            self.test_res.append(
                [
                    self.test_on,
                    img_name[0],
                    _metrics["psnr"].item(),
                    _metrics["ssim"].item(),
                    kwargs["used_time"],
                ]
            )
        if stage != "test":
            self.log_dict(
                res,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        return res

    def training_step(self, batch, batch_idx):
        res = self._shared_eval_step(batch, batch_idx, metrics=self.train_metrics)
        return {"loss": res["train-loss"]}

    def validation_step(self, batch, batch_idx):
        res = self._shared_eval_step(
            batch, batch_idx, metrics=self.val_metrics, stage="val"
        )

    def test_step(self, batch, batch_idx):
        res = self._shared_eval_step(
            batch, batch_idx, metrics=self.test_metrics, stage="test"
        )

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, logger=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, logger=True, prog_bar=True)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        # self.log_dict(metrics, logger=True)

    def start_test(self):
        self.test_res = []

    def save_test_res(self, save_path):
        data = pd.DataFrame(
            self.test_res, columns=["dataset", "image", "psnr", "ssim", "time"]
        )
        print(data.groupby(["dataset"]).mean(numeric_only=True))
        data.to_csv(save_path, index=False)
        self.test_res = []
        return data

