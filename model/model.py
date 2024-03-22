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

import torch
import torch.nn as nn
from einops import rearrange

from .sampling import build_sampling_model
from .utils import Head, Tail, Uformer, load_phi_psi


class CS_model(torch.nn.Module):
    def __init__(
        self,
        sr,
        blk_size=32,
        sampling_mode="constant",
        saliency_mode="sm",
        sr_base_ratio=3.0,
        dim=32,
        depths=[3, 5, 7, 7],
        num_heads=[1, 2, 4, 8],
        window_size=[8, 8, 4, 4],
        mlp_ratio=2.0,
        ## cs projection in transformer block
        cs_projection=True,
        cs_proj_by_channel=True,
        # fuse block
        fuse_type="concat",
        conv_after_fusion=False,
        cfg=None,
        **kwargs
    ):
        super().__init__()
        self.sr = sr
        self.blk_size = blk_size
        self.cfg = cfg

        self.sampling_model = build_sampling_model(
            sr=sr,
            blk_size=blk_size,
            sampling_mode=sampling_mode,
            saliency_mode=saliency_mode,
            sr_base_ratio=sr_base_ratio,
        )

        self.head = Head()
        self.transformer = Uformer(
            dim=dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            ## cs projection in transformer block
            blk_size=blk_size,
            cs_projection=cs_projection,
            cs_proj_by_channel=cs_proj_by_channel,
            # fuse block
            fuse_type=fuse_type,
            conv_after_fusion=conv_after_fusion,
            cfg=cfg
        )
        self.tail = Tail()

    def reconstruction(self, x0, res=None, **kwargs):
        x_re = self.head(x0)
        x_re = self.transformer(x_re, self.sampling_model, x0, res=res, **kwargs)
        x_re = self.tail(x_re) + x0
        return x_re

    def forward(self, x, **kwargs):
        res = self.sampling_model(x)
        res['x_re'] = self.reconstruction(res['x0'], res=res, **kwargs)
        return res

# + tags=["active-ipynb", "style-commentate"]
# model = CS_model(sr=0.1, args=None)
# x = torch.rand(2, 1, 256, 256)
# model(x)[1] - x
