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

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath

from .swin_transformer import BasicLayer, SwinTransformerBlock


# + tags=["active-ipynb", "style-activity"]
# from swin_transformer import BasicLayer, SwinTransformerBlock
# -

# # Measurement matrix

def Orthogonal_matrix(m: int, n: int) -> np.ndarray:
    """
    generate measurement matrix for CS model.
    Args:
        m: n * sr
        n: the original length of a vector
    Return:
        a matrix with shape (m, n)
    """

    def normalize(v):
        return v / np.sqrt(v.dot(v))

    np.random.seed(42)
    phi = np.random.normal(0, 1 / m, (m, n))

    # perform Gramm-Schmidt orthonormalization
    phi[0, :] = normalize(phi[0, :])
    for i in range(1, m):
        Ai = phi[i, :]
        for j in range(0, i):
            Aj = phi[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        phi[i, :] = normalize(Ai)

    return phi.astype(np.float32)


def load_phi_psi(blk_size, sr):
    if sr <= 0:
        matrix = Orthogonal_matrix(blk_size**2, blk_size**2)
    else:
        n = blk_size**2
        m = int(n * sr)
        matrix = Orthogonal_matrix(m, n)
    return matrix


# # Model modules

# ## head & Tail

class Head(nn.Module):
    def __init__(self, drop_path=0.1):
        r""" """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, 32, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.block1(x)
        x = x + self.drop_path(self.alpha * self.block2(x))
        return x


class Tail(nn.Module):
    def __init__(self, drop_path=0.1):
        r""" """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, 32, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.alpha * self.block1(x))
        x = self.block2(x)
        return x


# ## U-Net modules

# ### Downsample and Upsmaple

# +
class Downsample(nn.Module):
    def __init__(self, dim, factor=2):
        r"""

        down-sample a image: (B, C, H, W) -> (B, C//2, H, W) -> (B, 2C, H//2, W//2)

        Args:
            dim (int): the image channels
        """
        super().__init__()
        # print(f"Downsample layer:  input dim is {dim}, factor is {factor}")
        if factor == 1:
            self.down_sample = nn.Identity()
        else:
            self.down_sample = nn.Sequential(
                nn.Conv2d(dim, dim // factor, 3, padding=1, bias=False),
                Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=factor, t2=factor),
            )

    def forward(self, x):
        return self.down_sample(x)


class Upsample(nn.Module):
    def __init__(self, dim, factor=2):
        r"""

        up-sample a image: (B, 2C, H//2, W//2) -> (B, 4C, H//2, W//2) -> (B, C, H, W)

        Args:
            dim (int): the image channels
        """
        super().__init__()
        if factor == 1:
            self.up_sample = nn.Identity()
        else:
            self.up_sample = nn.Sequential(
                nn.Conv2d(dim, dim * factor, 3, padding=1, bias=False),
                Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=factor, t2=factor),
            )

    def forward(self, x):
        return self.up_sample(x)


# + tags=["style-student", "active-ipynb"]
# x = torch.rand(2, 32, 128, 128)
#
# down = Downsample(32)
# up = Upsample(64)
# up(down(x)).shape
# -

# ### Fusion block

class Fusion_block(nn.Module):
    def __init__(
        self, dim, fuse_type="concat", conv_after_fusion=False, drop_path=0.1, **kwargs
    ):
        r""" """
        super().__init__()

        self.type = fuse_type
        assert fuse_type in ["concat", "add"]
        if self.type == "concat":
            self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False)

        self.conv_after_fusion = conv_after_fusion
        if conv_after_fusion:
            alpha_0 = 1e-2
            self.alpha = nn.Parameter(
                alpha_0 * torch.ones((1, dim, 1, 1)), requires_grad=True
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.conv_block = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            )

    def forward(self, a, b):
        if self.type == "concat":
            x = torch.concat([a, b], dim=1)
            x = self.proj(x)
        else:
            x = a + b

        if self.conv_after_fusion:
            x = x + self.drop_path(self.alpha * self.conv_block(x))

        return x


# + tags=["active-ipynb", "style-student"]
# x = torch.rand(2, 32, 128, 128)
# y = torch.rand(2, 32, 128, 128)
# module = Fusion_block(dim=32, conv_after_fusion=True)
# module(x, y).shape
# -

# ### CS Projection

class CS_Projection(nn.Module):
    def __init__(
        self,
        downsample_factor=1,
        blk_size=32,
        cs_proj_by_channel=True,
        dim=32,
        cfg=None,
        **kwargs
    ):
        r""" """
        super().__init__()

        self.downsample_factor = downsample_factor
        if downsample_factor > 1:
            self.upsample = Rearrange(
                "b (c t1 t2) h w -> b c (h t1) (w t2)",
                t1=downsample_factor,
                t2=downsample_factor,
            )

            self.downsample = Rearrange(
                "b c (h t1) (w t2) -> b (c t1 t2) h w",
                t1=downsample_factor,
                t2=downsample_factor,
            )

        else:
            self.upsample = nn.Identity()
            self.downsample = nn.Identity()

        self.blk_size = blk_size
        self.cfg = cfg

        self.cs_proj_by_channel = cs_proj_by_channel
        if cs_proj_by_channel == False:
            self.down_channel = nn.Conv2d(
                dim // (downsample_factor**2), 1, 3, padding=1, bias=False
            )
            self.up_channel = nn.Conv2d(
                1, dim // (downsample_factor**2), 3, padding=1, bias=False
            )

        self.is_learnable_cs_proj_step = False
        if hasattr(cfg, "cs_proj_learnable_step") and cfg.cs_proj_learnable_step:
            self.is_learnable_cs_proj_step = True
            # print("Use learnable projection step")
            self.cs_proj_learnable_step = nn.Parameter(
                torch.zeros((1, dim // (downsample_factor**2), 1, 1)),
                requires_grad=True,
            )
            if cs_proj_by_channel == False:
                self.cs_proj_learnable_step = nn.Parameter(
                    torch.zeros((1, 1, 1, 1)), requires_grad=True
                )

    #     def sampling(self, x, phi):
    #         B, C, H, W = x.shape
    #         x_blocks = rearrange(
    #             x,
    #             "B C (h b1) (w b2) -> B h w (C b1 b2)",
    #             b1=self.blk_size,
    #             b2=self.blk_size,
    #             h=H // self.blk_size,
    #             w=W // self.blk_size,
    #         )
    #         y = torch.matmul(x_blocks, phi)
    #         return y

    #     def reconstruction(self, y, psi, x_shape):
    #         B, C, H, W = x_shape
    #         x0 = torch.matmul(y, psi)
    #         x0 = rearrange(
    #             x0,
    #             "B h w (C b1 b2) -> B C (h b1) (w b2)",
    #             b1=self.blk_size,
    #             b2=self.blk_size,
    #             C=1,
    #             h=H // self.blk_size,
    #             w=W // self.blk_size,
    #         )
    #         return x0

    def forward(self, x_i, sampling_model, x0, phi_T=False, res=None, **kwargs):
        """

        Args:
            x_i: (B, C, H, W)
            phi: (1024, m)
            psi: (m, 1024)
            x0: (B, 1, H, W)
        """

        # print("input shape is ", x_i.shape)
        x_i = self.upsample(x_i)
        B, C, H, W = x_i.shape
        # print("after upsample is ", x_i.shape)

        if "save_middle_results" in kwargs.keys() and kwargs["save_middle_results"]:
            self.middle_results = [x_i]

        if self.cs_proj_by_channel:
            x_i = rearrange(x_i, "b c h w  -> (b c) 1 h w")
            # x_t = self.reconstruction(self.sampling(x_i, phi), psi, x_shape=x_i.shape)
            # print(sampling_model)
            x_t = sampling_model.projection(x_i, phi_T=phi_T)
            x_t = rearrange(x_t, "(b c) 1 h w  -> b c h w", b=B, c=C)
            x_i = rearrange(x_i, "(b c) 1 h w  -> b c h w", b=B, c=C)
            # print(x_i.shape, x_t.shape, x0.shape)

            x0 = res["x0_T"] if phi_T else x0
            if not self.is_learnable_cs_proj_step:
                x_i = x_i - x_t + x0
            else:
                x_i = x_i + (x0 - x_t) / (1 + self.cs_proj_learnable_step)
        else:
            x_i = self.down_channel(x_i)  # (B, 1, H, W)
            # x_t = self.reconstruction(self.sampling(x_i, phi), psi, x_shape=x_i.shape)
            x_t = sampling_model.projection(x_i)
            x_i = x_i + (x0 - x_t) / (1 + self.cs_proj_learnable_step)
            x_i = self.up_channel(x_i)

        if "save_middle_results" in kwargs.keys() and kwargs["save_middle_results"]:
            self.middle_results.append(x_i)
        # print("before down sample is ", x_i.shape)
        x_i = self.downsample(x_i)
        # print("after down sample is ", x_i.shape)
        return x_i

# + tags=["style-student", "active-ipynb"]
# x = torch.rand(2, 32, 32, 32)
# module = CS_Projection(
#     downsample_factor=4, blk_size=32, dim=32, cs_proj_by_channel=False
# )
# phi = torch.rand(1024, 100)
# psi = torch.rand(100, 1024)
# x0 = torch.rand(2, 1, 128, 128)
# module(x, phi, psi, x0).shape
# -
# ### Transformer Block for CS


class CS_Transformer_blocks(nn.Module):
    def __init__(
        self,
        dim=32,
        input_resolution=128,
        depth=2,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        ## cs
        cs_projection=True,
        downsample_factor=1,
        blk_size=32,
        cs_proj_by_channel=True,
        cfg=None,
        **kwargs
    ):
        super().__init__()

        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )

        self.cs_projection = cs_projection
        if cs_projection:
            # print("Use cs projection in Transformer")
            self.cs_projection_block = CS_Projection(
                dim=dim,
                blk_size=blk_size,
                cs_proj_by_channel=cs_proj_by_channel,
                downsample_factor=downsample_factor,
                cfg=cfg,
            )

    def forward(self, x, sampling_model=None, x0=None, res=None, **kwargs):
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.blocks:
            x = block(x, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        if self.cs_projection:
            phi_T = True if self.cs_projection == 2 else False
            x = self.cs_projection_block(
                x, sampling_model, x0, phi_T=phi_T, res=res, **kwargs
            )
        return x


# + tags=["active-ipynb", "style-student"]
# x = torch.rand(2, 32, 128, 128)
# model = CS_Transformer_blocks(cs_projection=True, input_resolution=96)
# phi = torch.rand(1024, 100)
# psi = torch.rand(100, 1024)
# x0 = torch.rand(2, 1, 128, 128)
# model(x, phi, psi, x0).shape
# -

# ## U-Net

class Uformer(nn.Module):
    def __init__(
        self,
        dim=32,
        depths=[5, 5, 5, 5],
        num_heads=[1, 2, 4, 8],
        window_size=[8, 8, 4, 4],
        mlp_ratio=2.0,
        ## cs projection in transformer block
        blk_size=32,
        cs_projection=True,
        cs_proj_by_channel=True,
        # fuse block
        fuse_type="concat",
        conv_after_fusion=False,
        cfg=None,
        **kwargs
    ):
        r""" """
        super().__init__()

        self.down_samplers = nn.ModuleList(
            [Downsample(dim=dim * (2**i)) for i in range(3)]
        )
        self.up_samplers = nn.ModuleList(
            [Upsample(dim=dim * (2**i)) for i in range(1, 4)]
        )
        self.fuse_blocks = nn.ModuleList(
            [
                Fusion_block(
                    dim=dim * (2**i),
                    fuse_type=fuse_type,
                    conv_after_fusion=conv_after_fusion,
                )
                for i in range(3)
            ]
        )

        self.encoder_blocks = nn.ModuleList(
            [
                CS_Transformer_blocks(
                    dim=dim * (2**i),
                    depth=depths[i],
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    ## cs
                    cs_projection=cs_projection,
                    downsample_factor=2**i,
                    blk_size=blk_size,
                    cs_proj_by_channel=cs_proj_by_channel,
                    cfg=cfg,
                )
                for i in range(4)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                CS_Transformer_blocks(
                    dim=dim * (2**i),
                    depth=depths[i],
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    ## cs
                    cs_projection=cs_projection,
                    downsample_factor=2**i,
                    blk_size=blk_size,
                    cs_proj_by_channel=cs_proj_by_channel,
                    cfg=cfg,
                )
                for i in range(3)
            ]
        )

    def forward(self, x, sampling_model=None, x0=None, res=None, **kwargs):
        enc = []
        for i in range(3):
            x = self.encoder_blocks[i](x, sampling_model, x0, res=res, **kwargs)
            enc.append(x)
            x = self.down_samplers[i](x)

        x = self.encoder_blocks[3](x, sampling_model, x0, res=res, **kwargs)

        for i in [2, 1, 0]:
            x = self.fuse_blocks[i](self.up_samplers[i](x), enc[i])
            x = self.decoder_blocks[i](x, sampling_model, x0, res=res, **kwargs)
        return x

# + tags=["style-student", "active-ipynb"]
# x = torch.rand(2, 32, 128, 128)
# model = Uformer(dim=32)
# phi = torch.rand(1024, 100)
# psi = torch.rand(100, 1024)
# x0 = torch.rand(2, 1, 128, 128)
# model(x, phi, psi, x0).shape
