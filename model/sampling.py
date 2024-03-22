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

# +
import random

import torch
import torch.nn as nn
from .utils import load_phi_psi
from einops import rearrange
# -

from torchmetrics.functional import peak_signal_noise_ratio as PSNR

from .adaptive import Diff, Std, Traditional


# + tags=["active-ipynb", "style-activity"]
# from adaptive import Diff, Std, Traditional
# -

# # Code

# ## Base

class Base(nn.Module):
    def __init__(self, sr, blk_size=32, **kwargs):
        super().__init__()
        self.sr = sr
        self.blk_size = blk_size

    def space_to_depth(self, x):
        x_blocks = rearrange(
            x,
            "B C (h b1) (w b2) -> B h w (C b1 b2)",
            b1=self.blk_size,
            b2=self.blk_size,
        )
        return x_blocks

    def depth_to_space(self, x_blocks):
        x = rearrange(
            x_blocks,
            "B h w (C b1 b2) -> B C (h b1) (w b2)",
            b1=self.blk_size,
            b2=self.blk_size,
        )
        return x

    def setup_phi_psi(self, matrix):
        self.register_parameter("phi", nn.Parameter(torch.tensor(matrix.T)))
        self.register_parameter("psi", nn.Parameter(torch.tensor(matrix)))

    def sampling(self, x):
        raise NotImplementedError

    def init_reconstruction(self, y, phi_T=False, *kargs):
        if phi_T:
            x0 = torch.matmul(y, self.phi.T)
            # print('use transpose of phi.')
        else:
            x0 = torch.matmul(y, self.psi)
        x0 = self.depth_to_space(x0)
        return x0

    def forward(self, x):
        raise NotImplementedError

    def sampling_without_calc_saliency(self, x):
        raise NotImplementedError

    def projection(self, x):
        raise NotImplementedError


# ## Constant

class ConstantSampling(Base):
    def __init__(self, sr, blk_size=32, **kwargs):
        super().__init__(sr, blk_size)

        matrix = load_phi_psi(blk_size, sr)
        self.setup_phi_psi(matrix)

    def sampling(self, x, *kargs):
        x_blocks = self.space_to_depth(x)
        y = torch.matmul(x_blocks, self.phi)
        return y

    def forward(self, x):
        y = self.sampling(x)
        x0 = self.init_reconstruction(y)
        res = {"x0": x0, "y": y}
        return res

    def projection(self, x, *args, **kwargs):
        res = self.forward(x)
        return res["x0"]

    def sampling_without_calc_saliency(self, x):
        return self.sampling(x)


# ## Adaptive

class AdaptiveSampling(Base):
    def __init__(
        self, sr, blk_size=32, saliency_mode="sm", sr_base_ratio=3.0, **kwargs
    ):
        super().__init__(sr, blk_size)

        if sr == 999.0:
            matrix = load_phi_psi(blk_size, 1.0)
        else:
            matrix = load_phi_psi(blk_size, sr * 2)
        self.setup_phi_psi(matrix)

        self.sr_base_ratio = sr_base_ratio
        print(f"saliency mode is {saliency_mode}")
        if saliency_mode == "sm":
            self.saliency_model = Traditional(blk_size=blk_size)
        elif saliency_mode == "std":
            self.saliency_model = Std(blk_size=blk_size)
        elif saliency_mode == "diff":
            self.saliency_model = Diff(blk_size=blk_size)
        else:
            raise ValueError("Error, saliency mode should be `sm`, `std`")

        print(f"Adaptive sampling at sampling rate {sr}!!!")

    def get_sr_base_ratio(self, sr):
        if sr < 0.2:
            return 0.5
        else:
            return 0.3333

    def sampling(self, x):
        x_blocks = self.space_to_depth(x)

        SR = [1, 4, 10, 25, 50]
        if self.sr == 999.0:
            # sr = random.choice(SR) / 100
            sr = random.randint(1, 50) / 100
        else:
            sr = self.sr

        nB = int(self.blk_size**2 * sr)
        sr_base_ratio = self.get_sr_base_ratio(sr)
        self.saliency_model.sr_base_ratio = sr_base_ratio

        nB_init = int(nB * sr_base_ratio)
        y_init = torch.matmul(x_blocks, self.phi[:, 0:nB_init])
        
        if self.saliency_model.operate_on == 'img':
            x0_init = torch.matmul(y_init, self.psi[0:nB_init, :])
            x0_init = self.depth_to_space(x0_init)
            self.y_mask = self.saliency_model(x0_init, sr, mask=True)
        elif self.saliency_model.operate_on == 'measurement':
            self.y_mask = self.saliency_model(y_init, sr, mask=True)
            x0_init = None
            

        # print(f"sampling ratio is {sr}, sr_base_ratio is {sr_base_ratio}, nB is {nB}",\
        # "Init psnr is ", PSNR(x[0], x0_init[0]).cpu().detach().numpy())
        # print(torch.sum(self.y_mask!=0, dim=-1))

        y_next = torch.matmul(x_blocks, self.phi[:, nB_init:])
        self.y_mask = self.y_mask[:, :, :, : self.phi.shape[-1]]
        y = torch.concat([y_init, y_next], dim=-1) * self.y_mask
        return y, x0_init

    def forward(self, x):
        y, x0_init = self.sampling(x)
        x0 = self.init_reconstruction(y)
        x0_T = self.init_reconstruction(y, phi_T=True)
        res = {"x0": x0, "y": y, "y_mask": self.y_mask, "x0_init": x0_init, 'x0_T':x0_T}
        return res

    def sampling_without_calc_saliency(self, x):
        x_blocks = self.space_to_depth(x)
        y = torch.matmul(x_blocks, self.phi) * self.y_mask
        return y

    def projection(self, x, phi_T=False):
        """
        Args:
            x: '(B C) 1 H W'
        """
        x_blocks = self.space_to_depth(x)
        y = torch.matmul(x_blocks, self.phi)
        y = rearrange(y, "(b c) x y z -> c b x y z", b=self.y_mask.shape[0])
        y = y * self.y_mask
        y = rearrange(y, "c b x y z -> (b c) x y z")
        x0 = self.init_reconstruction(y, phi_T=phi_T)
        return x0


# # Build Sampling Model

def build_sampling_model(
    sr, blk_size=32, sampling_mode="constant", saliency_mode="sm", sr_base_ratio=3.0
):
    if sampling_mode == "constant":
        sampling_model = ConstantSampling(sr=sr, blk_size=blk_size)
    elif sampling_mode == "adaptive":
        sampling_model = AdaptiveSampling(
            sr=sr,
            blk_size=blk_size,
            saliency_mode=saliency_mode,
        )
    else:
        raise ValueError(f"Error, sampling_mod is {sampling_mode}")
    return sampling_model
