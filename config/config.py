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

# +
import os

from yacs.config import CfgNode as ConfigurationNode
# -

# # 默认配置

DATA_PATH = {
    "coco": "/usr/local/ay_data/dataset/1-CS/COCO",
    "set5": "/usr/local/ay_data/dataset/1-CS/Set5",
    "set14": "/usr/local/ay_data/dataset/1-CS/Set14",
    "bsd68": "/usr/local/ay_data/dataset/1-CS/BSD68",
    "bsd100": "/usr/local/ay_data/dataset/1-CS/BSD100",
    "cbsd68": "/usr/local/ay_data/dataset/1-CS/CBSD68",
    "train400": "/usr/local/ay_data/dataset/1-CS/train400",
    "urban100": "/usr/local/ay_data/dataset/1-CS/Urban100",
}
_paths = ConfigurationNode(DATA_PATH)

# +
_dataset = ConfigurationNode()
_dataset.trainset = "coco"  # dataset name for training
_dataset.train_num = 40000  # the nubmer of used images in the train seet
_dataset.n_samples_per_epoch = 40000  # the nubmer of images in each train epoch
_dataset.train_size = 256  # the image size of the training images
_dataset.batch_size = 8  # batch size in training

_dataset.val_rate = 100.0  # if > 0, the trainset will split a part for validation
_dataset.testset = ["set5", "set14", "set11", "bsd100", "urban100"]  # the test datasets
# -

_augmentation = ConfigurationNode()
_augmentation.flip = True
_augmentation.rotate = True

_model = ConfigurationNode()
_model.lr = 0.0001
_model.blk_size = 32
_model.sr = 0.1
_model.sampling_mode = "constant"
_model.saliency_mode = "sm"
_model.sr_base_ratio = 3.0
_model.epochs = 100
_model.earlystop = 3
_model.dim = 32
_model.depths = [3, 5, 7, 7]
_model.num_heads = [1, 2, 4, 8]
_model.window_size = [8, 8, 4, 4]
_model.mlp_ratio = 2.0
_model.cs_projection = True
_model.cs_proj_by_channel = True
_model.cs_proj_learnable_step = False
_model.fuse_type = "concat"
_model.loss_type = "l1-l2-l3"
_model.conv_after_fusion = False
_model.train_phi = False

__C = ConfigurationNode()
__C.Paths = _paths
__C.Dataset = _dataset
__C.Augmentation = _augmentation
__C.Model = _model


def get_cfg_defaults(cfg_file=None, ablation=""):
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    res = __C.clone()

    if cfg_file is not None:
        if not cfg_file.endswith(".yaml"):
            cfg_file += ".yaml"

        dataset_file_path = os.path.join(os.path.split(cfg_file)[0], "dataset.yaml")
        if os.path.exists(dataset_file_path):
            res.merge_from_file(dataset_file_path)
            print("load dataset yaml in ", dataset_file_path)

        model_file_path = os.path.join(os.path.split(cfg_file)[0], "model.yaml")
        if os.path.exists(model_file_path):
            res.merge_from_file(model_file_path)
            print("load model yaml in ", model_file_path)

        #         if ablation != '':
        #             ablation_path = os.path.join(os.path.split(cfg_file)[0], "%s.yaml"%ablation)
        #             res.merge_from_file(ablation_path)
        #             print("load ablation yaml in ", ablation_path)
        res.merge_from_file(cfg_file)

    assert res.Dataset.trainset in DATA_PATH.keys()
    return res
