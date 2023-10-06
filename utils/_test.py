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

def test_dataset_constant(model, trainer, log_dir):
    model.start_test()

    for test_set in ["set5", "set11", "set14", "bsd100", "urban100"]:
    # for test_set in ["set5"]:
        test_dl = get_testset(test_set)
        model.test_on = test_set
        trainer.test(model, test_dl)
    model.save_test_res(os.path.join(log_dir, "test.csv"))


def test_dataset_adaptive(model, trainer, log_dir):
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
            trainer.test(model, test_dl)
        model.save_test_res(os.path.join(log_dir, "test_%d.csv"%sr))


# ## 在几张图像上进行测试

def test_images_scalable(model, trainer, name):
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
            trainer.test(model, test_dl)


def test_images_constant(model, trainer, name):
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
        trainer.test(model, test_dl)


# ## 测试时间复杂度

def test_time_complexity(model, trainer, name):
    
    # _sr = 0.25
    # model.sr = _sr
    # model.model.sr = _sr
    # model.model.sampling_model.sr = _sr
    
    
    model.start_test()
    if hasattr(model, 'set_sr'):
        model.set_sr(0.25)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params, total_trainable_params)

    print(model.device)

    inp = torch.randn(1, 1, 256, 256, device=model.device)
    ms = get_module_summary(model.model, module_args=(inp,))
    flops = ms.flops_forward / 1e9
    num_para = ms.num_parameters / 1e6
    
    print(f'FLOPs is {flops} G, parameters is {num_para} M')
    flops_csv = '0-实验结果/datas/time/flops_para.csv'
    if not os.path.exists(flops_csv):
        data = pd.DataFrame(columns=['model', 'flop', 'para'])
    else:
        data = pd.read_csv(flops_csv, index_col=0)
    data.loc[name] = [name, flops, num_para]
    data.to_csv(flops_csv)
    
    
    
    for test_set in ["set1024"]:
        test_dl = get_testset(test_set.lower())
        model.test_on = test_set
        trainer.test(model, test_dl)
        model.test_res = [] 
        trainer.test(model, test_dl)
    model.save_test_res(os.path.join('0-实验结果/datas/time', f"{name}.csv"))
