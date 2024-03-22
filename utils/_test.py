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

import os
from data import get_testset, get_trainset
import torch
# from .flop_counter import FlopCounterMode


import pandas as pd
import os

# +
import torch
# from torcheval.tools.module_summary import get_module_summary

# 

# from pytorch_lightning.demos.boring_classes import BoringModel

# model = BoringModel()
# ms = get_module_summary(model)
# print(ms)
# summary = get_module_summary(model, module_args=(torch.randn(2, 32),))
# print(summary.flops_forward, summary.flops_backward, summary)
# -

def test_dataset_constant(model, trainer, log_dir, ckpt_path=None):
    model.start_test()

    for test_set in ["set5", "set11", "set14", "bsd100", "urban100"]:
    # for test_set in ["set5"]:
        test_dl = get_testset(test_set)
        model.test_on = test_set
        trainer.test(model, test_dl, ckpt_path=ckpt_path)
    model.save_test_res(os.path.join(log_dir, "test.csv"))


def test_dataset_adaptive(model, trainer, log_dir, ckpt_path=None):
    model.is_training = False
    model.start_test()

    for sr in [1, 4, 10, 25, 50]:
        _sr = sr / 100
        model.set_sr(_sr)
        # model.sr = _sr
        # model.model.sr = _sr
        # model.model.sampling_model.sr = _sr
        
        print("Test on sr %.2f"%(_sr))
        for test_set in ["set5", "set11", "set14", "bsd100", "urban100"]:
            test_dl = get_testset(test_set)
            model.test_on = test_set
            trainer.test(model, test_dl, ckpt_path=ckpt_path)
        model.save_test_res(os.path.join(log_dir, "test_%d.csv"%sr))


# ## 在几张图像上进行测试

def test_images_scalable(model, trainer, name, ckpt_path=None):
    model.start_test()

    for sr in [4, 10]:
        _sr = sr / 100
        model.set_sr(_sr)
        # model.sr = _sr
        # model.model.sr = _sr
        # model.model.sampling_model.sr = _sr
        
        for test_set in ["test_for_visual_comparison"]:
            test_dl = get_testset(test_set)
            model.test_on = test_set
            model.save_test_image = True
            model.save_dir = '0-实验结果/pics/visual_comparsion'
            model.model_name = name
            trainer.test(model, test_dl, ckpt_path=ckpt_path)


def test_images_constant(model, trainer, name, ckpt_path=None):
    model.start_test()


    if hasattr(model.model, 'sampling_model') and '+' in name:
        test_images_scalable(model, trainer, name)
        return
    
    for test_set in ["test_for_visual_comparison"]:
        test_dl = get_testset(test_set)
        model.test_on = test_set
        model.save_test_image = True
        model.save_dir = '0-实验结果/pics/visual_comparsion'
        model.model_name = name
        trainer.test(model, test_dl, ckpt_path=ckpt_path)



