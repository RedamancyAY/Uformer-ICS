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

# +
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme



def Color_progress_bar():
    return RichProgressBar(
        theme=RichProgressBarTheme(
            description="#008000",
            progress_bar="#0000ff",
            progress_bar_finished="#00ff00",
            progress_bar_pulse="red",
            batch_progress="bold #00ff00 on #ff0000",
            time="#ff0000",
            processing_speed="#0000ff",
            metrics="bold #ff0000 on #00ff00",
        )
    )
