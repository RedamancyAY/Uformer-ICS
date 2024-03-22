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

import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule


class EarlyStop(pl.callbacks.EarlyStopping):
    def __init__(self, min_epochs, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)


class EarlyStoppingLR(Callback):
    """Early stop model training when the LR is lower than threshold."""

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True

# + tags=["active-ipynb", "style-student"]
# s = EarlyStop(min_epochs=10, monitor='psnr')
