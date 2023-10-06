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
import math

import torch
import torch_dct
from einops import rearrange
from kornia.filters import gaussian_blur2d


# + tags=[]
class Base(torch.nn.Module):
    def __init__(self, blk_size=32, sr_base_ratio=3.0, **kwargs):
        super().__init__()
        self.blk_size = blk_size
        self.sr_base_ratio = sr_base_ratio if sr_base_ratio < 1.0 else 1.0/sr_base_ratio

        self.operate_on = "img"  # 'img' or 'measurement'

        self.register_buffer(
            "nB_base_matrix", torch.arange(0, self.blk_size**2)[None, None, None, :]
        )

    def to_4D(self, img):
        n_dims = len(img.shape)
        if n_dims == 2:
            return img[None, None, :, :]
        if n_dims == 3:
            return img[None, ...]

    def get_block_saliency(self, s):
        raise NotImplementedError()

    def initial_assign(self, sr, C, H, W):
        max_nB = self.blk_size**2 * C
        h, w = H // self.blk_size, W // self.blk_size

        total = math.floor(sr * H * W * C)
        basic = math.ceil(sr * self.sr_base_ratio * max_nB)
        rest = total - basic * (h * w)
        
        # print(sr, h ,w , total, basic, rest)
        
        assert basic < total and rest > h * w

        # max_ratio = (max_nB - basic) / rest
        max_ratio = (int(max_nB * sr * 2) - basic) / rest

        return max_nB, total, basic, rest, max_ratio

    def adjust_si(self, si, max_ratio):
        """调整图像块的 ``saliency info``， 防止过大

        Args:
            si: the saliency info for every block
            max_ratio: the maximum value for every block

        Returns:
            the adjusted saliency info
        """
        # print('adjust si')
        
        a = torch.where(si > max_ratio, max_ratio, si)
        old = torch.sum(torch.where(si < max_ratio, si, 0), dim=[1, 2], keepdim=True)
        new = (
            1.0
            - torch.sum((a == max_ratio).type(torch.float32), dim=[1, 2], keepdim=True)
            * max_ratio
        )
        si = torch.where(a < max_ratio, a * new / old, a)
        return si

    def saliency(self):
        raise NotImplementedError()

    def forward(self, x, sr, mask=False):
        # these two function must be implemented
        saliency = self.saliency(x)
        saliency = self.get_block_saliency(saliency)

        # saliency = torch.sum(saliency, dim=1) # (B, H, W)
        saliency = saliency / torch.sum(saliency, dim=[1, 2], keepdim=True)

        if self.operate_on == "measurement":
            B, h, w, l = x.shape
            C, H, W = 1, h * self.blk_size, w * self.blk_size
        else:
            B, C, H, W = x.shape
        max_nB, total, basic, rest, max_ratio = self.initial_assign(sr, C, H, W)
        if torch.max(saliency) > max_ratio:
            saliency = self.adjust_si(saliency, max_ratio)
        nB = basic + torch.floor(saliency * rest)

        # print(basic, rest, saliency, nB)
        if mask:
            return self.nB2mask(nB)
        else:
            return nB

    def nB2mask(self, nB):
        x = self.nB_base_matrix * nB[..., None]
        mask = torch.where(x >= nB[..., None] ** 2, 0, 1)
        return mask.to(torch.float32)


# -

# # Traditional 

# ## saliency map

# + tags=[]
class Traditional(Base):
    def __init__(self, blk_size=32, sr_base_ratio=3.0):
        super().__init__(blk_size=blk_size, sr_base_ratio=sr_base_ratio)

    def saliency(self, x):
        assert len(x.shape) == 4, "the input should be (B, C, H, W)"

        x = x.type(torch.float64)
        x = torch_dct.dct_2d(x)
        P = torch.sign(x)
        F = torch.abs(torch_dct.idct_2d(P))
        S = gaussian_blur2d(F**2, (11.0, 11.0), (3.0, 3.0))
        S = S.type(torch.float32)
        return S

    def get_block_saliency(self, s):
        x = rearrange(
            s,
            "B C (h b1) (w b2) -> B h w (C b1 b2)",
            b1=self.blk_size,
            b2=self.blk_size,
        )
        return torch.sum(x, dim=-1)


# -

# ## Std

# + tags=[]
class Std(Base):
    def __init__(self, blk_size=32, sr_base_ratio=3.0):
        super().__init__(blk_size=blk_size, sr_base_ratio=sr_base_ratio)

    def saliency(self, x):
        x_blocks = rearrange(
            x,
            "B C (h b1) (w b2) -> B h w (C b1 b2)",
            b1=self.blk_size,
            b2=self.blk_size,
        )
        std = torch.std(x_blocks, dim=-1)
        return std

    def get_block_saliency(self, s):
        return s


# + tags=["active-ipynb", "style-student"]
# x = torch.randn(2, 1, 128, 128)
# module = Traditional(blk_size=32, sr_base_ratio=2.0)
# nB = module(x, 0.5)
# print(nB, torch.sum(nB, dim=[1, 2]) / x.shape[-2] / x.shape[-1])

# + tags=["active-ipynb", "style-student"]
# mask = module.nB2mask(nB)
#
# torch.sum(mask, dim=-1) - nB
# -

# ## Diff

# + tags=[]
class Diff(Base):
    def __init__(self, blk_size=32, sr_base_ratio=3.0):
        super().__init__(blk_size=blk_size, sr_base_ratio=sr_base_ratio)

        self.pad = torch.nn.ReflectionPad2d(1)
        self.operate_on = "measurement"

    def saliency(self, x):
        """
        Args:
            x: (B, h, w, L)
        Return:
            diff: (B, h, w, L)
        """
        x = rearrange(x, "b h w l -> b l h w")
        h, w = x.shape[-2], x.shape[-1]
        x2 = self.pad(x)
        x3 = torch.zeros_like(x)

        for i in range(3):
            for j in range(3):
                # print(x.shape, x2[:, :, i : i + h, j : j + w].shape)
                x3 = x3 + (x.abs() - x2[:, :, i : i + h, j : j + w].abs())
        x3 = x3 / 8
        x3 = rearrange(x3, "b l h w -> b h w l")
        saliency = torch.mean(x3.abs(), dim=-1)
        return saliency

    def get_block_saliency(self, s):
        return s

# + tags=["active-ipynb", "style-student"]
# y = torch.rand(2, 3, 3, 3)
# module = Diff(blk_size=32, sr_base_ratio=2.0)
# nB = module(y, 0.5)
# print(nB)
# mask = module.nB2mask(nB)
#
# torch.sum(mask, dim=-1) - nB
