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

# # Import packages

# +
import argparse
import os
import random
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from rich.console import Console
from torch.utils.data import DataLoader

# +
from config import get_cfg_defaults
from models import (
    AMP_Net,
    AMS_Net_lit,
    CSformer_lit,
    CSNetPlus_lit,
    DPC_DUN_lit,
    ISTA_Net_pp_lit,
    OCTUF_lit,
    SCSNet_lit,
    TCS_Net_lit,
    TransCS_lit,
)
from models.ours.lit_model import Uformer_ICS
from utils import (
    backup_logger_file,
    test_dataset_adaptive,
    test_dataset_constant,
    test_images_constant,
    test_time_complexity,
)

from data import get_testset, get_trainset
# -

from ay.torch.lightning.callbacks import (
    ChangeImageSizeAfterEpochs,
    EarlyStop,
    color_progress_bar,
)

# # Initialization

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")


# # Training

# ## Parse Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="CSNet+/coco")
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--test_time", type=int, default=0)
    parser.add_argument("--test_imgs", type=int, default=0)
    parser.add_argument("--version", type=int, default=-1)
    parser.add_argument("--sr", type=int, default=0.01)
    args = parser.parse_args()
    return args


def get_train_val(cfg):
    train_val_dl = get_trainset(cfg_ds=cfg.Dataset, cfg_aug=cfg.Augmentation)
    train_dl = train_val_dl["train"]
    val_dl = train_val_dl["val"]
    return train_dl, val_dl


def get_model(cfg):
    if args.cfg.startswith("CSNet+"):
        model = CSNetPlus_lit(sr=cfg.Model.sr, blk_size=32)
    elif args.cfg.startswith("AMP"):
        model = AMP_Net(sr=cfg.Model.sr, blk_size=32)
    elif args.cfg.startswith("TCS"):
        model = TCS_Net_lit(sr=cfg.Model.sr, blk_size=32)
    elif args.cfg.startswith("TransCS"):
        model = TransCS_lit(sr=cfg.Model.sr, blk_size=32)
    elif args.cfg.startswith("OCTUF"):
        model = OCTUF_lit(sr=cfg.Model.sr, blk_size=32)
    elif args.cfg.startswith("Ours"):
        print(cfg.Model)
        model = Uformer_ICS(sr=cfg.Model.sr, blk_size=32, args=cfg.Model)
    elif args.cfg.startswith("AMS_Net"):
        model = AMS_Net_lit(sr=cfg.Model.sr, blk_size=16, args=cfg.Model)
    elif args.cfg.startswith("SCSNet"):
        model = SCSNet_lit(sr=cfg.Model.sr, blk_size=32)
        model.model.is_training = True
    elif args.cfg.startswith("ISTA"):
        model = ISTA_Net_pp_lit(sr=cfg.Model.sr, blk_size=32)
        model.model.is_training = True
    elif args.cfg.startswith("CSformer"):
        model = CSformer_lit(sr=cfg.Model.sr, blk_size=64)
    elif args.cfg.startswith("DPC_DUN"):
        model = DPC_DUN_lit(sr=cfg.Model.sr, blk_size=32, args=cfg.Model)
    return model


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults(cfg_file=os.path.join("config/experiments", args.cfg))

    train_dl, val_dl = get_train_val(cfg)

    callbacks = [
        color_progress_bar,
        ModelCheckpoint(dirpath=None, save_top_k=1, monitor="val-psnr", mode="max"),
        EarlyStop(
            min_epochs=50 if not "999" in args.cfg else 75,
            monitor="val-psnr",
            min_delta=0.01,
            patience=cfg.Model.earlystop,
            mode="max",
            verbose=True,
        ),
        # ChangeImageSizeAfterEpochs(
        #     min_epochs=9, datasets=[train_dl.dataset], new_img_size=96
        # ),
    ]

    model = get_model(cfg)

    CS_root_dir = "/usr/local/ay_data/1-model_save/3-CS"
    args.version = 0 if (args.version == -1 and args.test) else args.version
    trainer = pl.Trainer(
        min_epochs=50 if cfg.Model.epochs > 50 else cfg.Model.epochs,
        max_epochs=cfg.Model.epochs,
        accelerator="gpu",
        devices=args.gpu,
        logger=pl.loggers.CSVLogger(
            save_dir=CS_root_dir,
            name=args.cfg,
            version=args.version if args.version != -1 else None,
        ),
        default_root_dir=CS_root_dir,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    log_dir = trainer.logger.log_dir
    Console().print(
        "[on #00ff00][#ff3300]logger path : %s[/#ff3300][/on #00ff00]" % log_dir
    )

    if args.test == 0:
        # test_time_complexity(model, trainer, name=args.cfg.split("/")[0])

        if args.resume:
            checkpoint = os.path.join(log_dir, "checkpoints")
            ckpt_path = os.path.join(checkpoint, os.listdir(checkpoint)[0])
            print("Resume from ", ckpt_path)
            trainer.fit(model, train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_dl, val_dataloaders=val_dl)

    else:
        backup_logger_file(log_dir)
        checkpoint = os.path.join(log_dir, "checkpoints")
        ckpt_path = os.path.join(checkpoint, os.listdir(checkpoint)[0])
        print(ckpt_path)
        model = model.load_from_checkpoint(
            ckpt_path, sr=cfg.Model.sr, blk_size=cfg.Model.blk_size, args=cfg.Model
        )
        # model = model.load_from_checkpoint(ckpt_path)
        model.eval()
        model.start_test()

        if args.test_time:
            test_time_complexity(
                model,
                trainer,
                name=args.cfg.split("/")[0]
                if "999" not in args.cfg
                else args.cfg.split("/")[0] + "+",
            )
        elif args.test_imgs:
            name = args.cfg.split("/")[0]
            name = name + "+" if "999" in args.cfg else name
            test_images_constant(model, trainer, name=name)
        else:
            if "999" in args.cfg:
                test_dataset_adaptive(model=model, trainer=trainer, log_dir=log_dir)
            else:
                test_dataset_constant(model=model, trainer=trainer, log_dir=log_dir)
