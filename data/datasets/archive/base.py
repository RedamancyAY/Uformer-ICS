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
import os

import pandas as pd
import torch
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


# -

class CS_BASE(torch.utils.data.Dataset):
    '''
        the base dataset for CS training and testing.
    '''
    def __init__(self):
        '''
            Args:
                root_path: the root path for the COCO dataset
                generate_meta: whether to generate 'dataset_info.csv' in the `root_path`.\
                    if it doesn't exist, will generate it; if it exists and generate_meta \
                    is true, will re-generate it.
        '''
        super().__init__()
        

    def get_metadata(self, root_path, generate_meta=False):
        '''
            get metadata for each image in the root_path.
        '''
        
        data_path = os.path.join(root_path, 'dataset_info.csv')
        
        if os.path.exists(data_path) and generate_meta == False:
            data = pd.read_csv(data_path)
            return data
        
        print(f"There is no dataset_info.csv in the path {root_path}, start to generate it.")
        
        img_names = sorted(os.listdir(root_path))
        img_names = [name for name in img_names if name.endswith("jpg")]
        data = pd.DataFrame()
        data["name"] = img_names

        def _metadata(img_path):
            img = Image.open(os.path.join(self.root_path, img_path))
            width, height = img.size
            n_channels = len(img.getbands())
            return n_channels, width, height

        tqdm.pandas(desc='    Start to extract metadata: (C, H, W)') 
        data[["n_channels", "width", "height"]] = data.progress_apply(
            lambda x: _metadata(x["name"]), axis=1, result_type="expand"
        )
        
        indices = np.arange(len(data))
        np.random.seed(42)
        np.random.shuffle(indices)
        data['id'] = indices
        
        data.to_csv(data_path, index=False)
        return data
