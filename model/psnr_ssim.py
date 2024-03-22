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

# + tags=[]
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# -


# + tags=["active-ipynb"]
# from functional import compute_psnr, compute_ssim

# + tags=[]
class PSNR(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("current", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")
        self.metrics = torch.tensor([])
        self.method = PeakSignalNoiseRatio(
            data_range=255.0, reduction="none", dim=[1, 2, 3]
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (B, C, H, W)
            target: (B, C, H, W)
        """
        if not preds.is_cpu:
            preds = preds.cpu().detach()
            target = target.cpu().detach()

        res = self.method(preds, target)

        self.current = torch.mean(res)
        if res.dim() > 0:
            self.metrics = torch.concat([self.metrics, res])
        else:
            self.metrics = torch.concat([self.metrics, torch.tensor([res])])

    def compute(self):
        self.total = torch.mean(self.metrics)
        self.metrics = torch.tensor([])
        return self.total


# + tags=[] jupyter={"source_hidden": true}
class SSIM(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("current", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")
        self.metrics = torch.tensor([])
        self.method = StructuralSimilarityIndexMeasure(data_range=255.0, reduction="none")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (B, C, H, W)
            target: (B, C, H, W)
        """
        if not preds.is_cpu:
            preds = preds.cpu().detach()
            target = target.cpu().detach()

        res = self.method(preds, target)

        self.current = torch.mean(res)
        if res.dim() > 0:
            self.metrics = torch.concat([self.metrics, res])
        else:
            self.metrics = torch.concat([self.metrics, torch.tensor([res])])

    def compute(self):
        self.total = torch.mean(self.metrics)
        self.metrics = torch.tensor([])
        return self.total


# + tags=[]
class PSNR_SSIM(object):
    def __init__(
        self, prefix=''
    ):
        super().__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.prefix=prefix

    def update(self, preds, target):
        psnr = self.psnr.update(preds, target)
        ssim = self.ssim.update(preds, target)
        return {"psnr": self.psnr.current, "ssim": self.ssim.current}

    def compute(self):
        res = {}
        res[self.prefix + "-psnr"] = self.psnr.compute()
        res[self.prefix + "-ssim"] = self.ssim.compute()
        return res

# + tags=["active-ipynb", "style-student"]
# my_metrics = PSNR_SSIM(prefix='test')
# for i in range(3):
#     target = torch.randint(0, 256, (i+1, 1, 256, 256)).type(torch.float32)
#     preds = torch.randint(0, 256, (i+1, 1, 256, 256)).type(torch.float32)
#     my_metrics.update(preds, target)
#
# print(my_metrics.psnr.metrics, my_metrics.ssim.metrics)
# res = my_metrics.compute()
# print(res)

# + tags=["active-ipynb"]
# compute_psnr(preds.numpy()[0], target.numpy()[0])
