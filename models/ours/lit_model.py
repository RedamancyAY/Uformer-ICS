# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay.torch.cs.lit_model import ConstantCS
from einops import rearrange

from .model import CS_model

from ay.torch.metrics import PSNR_SSIM
from ay.torch.optim import Optimizers_with_selective_weight_decay


class Uformer_ICS(ConstantCS):
    def __init__(self, sr, blk_size=32, args=None):
        model = CS_model(
            sr=sr,
            blk_size=blk_size,
            sampling_mode=args.sampling_mode,
            saliency_mode=args.saliency_mode,
            sr_base_ratio=args.sr_base_ratio,
            dim=args.dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=2.0,
            ## cs projection in transformer block
            cs_projection=args.cs_projection,
            cs_proj_by_channel=args.cs_proj_by_channel,
            # fuse block
            fuse_type=args.fuse_type,
            conv_after_fusion=args.conv_after_fusion,
            cfg=args,
        )

        super().__init__(model=model, sr=sr, blk_size=blk_size, args=args)

        self.f_loss = nn.MSELoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        model_params = self.model.parameters()
        # optimizer = torch.optim.Adam(model_params, lr=0.0001, weight_decay=0.05)
        optimizer = Optimizers_with_selective_weight_decay(
            model=self.model,
            lr=0.0001,
            weight_decay=0.01,
            optimizer="AdamW",
            debug=False,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            threshold=0.01,
            threshold_mode="abs",
            min_lr=0.000001,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val-psnr",
            },
        }

    def get_loss(self, model_res, target):
        l1 = self.f_loss(target, model_res["x_re"])
        _y = self.model.sampling_model.sampling_without_calc_saliency(model_res["x_re"])
        l2 = self.f_loss(model_res["y"], _y)
        l3 = self.f_loss(target, model_res["x0"])
        if self.model.cfg.loss_type == 'l1-l2-l3':
            loss = l1 + 0.1 * l2 + l3
        elif self.model.cfg.loss_type == 'no-l2':
            loss = l1 + l3
        elif self.model.cfg.loss_type == 'no-l2-l3-0.1':
            loss = l1 + 0.1 * l3
        elif self.model.cfg.loss_type in ['no-l3', 'no-l3*']:
            loss = l1 + 0.1 * l2
        elif self.model.cfg.loss_type == 'no-l2-l3':
            loss = l1
        elif self.model.cfg.loss_type == 'l3-0.1':
            loss = l1 + 0.1 * l2 + 0.1 * l3
        elif self.model.cfg.loss_type == 'l3-0.5':
            loss = l1 + 0.1 * l2 + 0.5 * l3
        else:
            raise ValueError('Error loss type!!!!')
            
        if "x0_init" in model_res.keys() and model_res['x0_init'] is not None:
            if not self.model.cfg.loss_type == 'no-l3*':
                loss = loss + 0.1 * self.f_loss(target, model_res["x0_init"])
        return loss
    
    def set_sr(self, _sr):
        self.sr = _sr
        self.model.sr = _sr
        self.model.sampling_model.sr = _sr
