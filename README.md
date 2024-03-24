# Uformer-ICS

This repository is the codes of the [Paper](https://ieeexplore.ieee.org/abstract/document/10323186).


## Prepare datasets

We use the COCO datasets for training and use the set5, set11, set14, bsd100, and urban100 for testing. Please download these datasets and specify the absolute path of these datasets in `data/make_dataset.py`:
```python
DATA_PATH = {
    "coco": "path_of/COCO",
    "set5": "path_of/Set5",
    "set11": "path_of/Set11",
    "set14": "path_of/Set14",
    "bsd100": "path_of/BSD100",
    "urban100": "path_of/Urban100",
    # 'set1024': "path_of/Set1024", # used for testing time complexity
    # 'test_for_visual_comparison':"path_of/test_for_visual_comparison" # used for visual comparison,
}
```
## Python Environment

```bash
pip install torch pytorch_lightning
pip install timm
pip install einops
pip install pandas
pip install pandas
pip install scipy
pip install torch_dct
pip install kornia
pip install yacs
```

## Training and Testing


in `trian.py`, we define `CS_root_dir`  to  save the all the training results. For example, if you train our method with sr (1, 4, 10, 25, 50), then the training results are saved in:
```
- CS_root_dir
    - Ours
        - coco
            - 1
                - version_0
                - version_1
                - ...
            - 4
                - version_0
                - version_1
                - ...
```


### Training scripts 

```bash
# train Uformer-ICS without adaptive sampling
python train.py --gpu 0 --cfg "Ours/coco/1";\
python train.py --gpu 0 --cfg "Ours/coco/4";\
python train.py --gpu 0 --cfg "Ours/coco/10";\
python train.py --gpu 0 --cfg "Ours/coco/25";\
python train.py --gpu 0 --cfg "Ours/coco/50";\

# train Uformer-ICS without adaptive sampling
python train.py --gpu 0 --cfg "Ours_adaptive/coco/1";\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/4";\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/10";\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/25";\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/50";\

# train Uformer-ICS+
python train.py --gpu 0 --cfg "Ours_adaptive/coco/999";
```

### Testing scripts

Please prepare ["set5", "set11", "set14", "bsd100", "urban100"] for evaluation, and give their path at `DATA_PATH`. If you want to test other datasets, please give their path at `DATA_PATH` and change the test list in `test_dataset_constant`, `test_dataset_adaptive` in the `utils/_test.py`.


```bash
# test Uformer-ICS without adaptive sampling
python train.py --gpu 0 --cfg "Ours/coco/1" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours/coco/4" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours/coco/10" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours/coco/25" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours/coco/50" --test 1 --version 0;\

# test Uformer-ICS without adaptive sampling
python train.py --gpu 0 --cfg "Ours_adaptive/coco/1" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/4" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/10" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/25" --test 1 --version 0;\
python train.py --gpu 0 --cfg "Ours_adaptive/coco/50" --test 1 --version 0;\

# test Uformer-ICS+
python train.py --gpu 0 --cfg "Ours_adaptive/coco/999"  --test 1 --version 0;
```

# Citation

If the code is used in your research, please Star our repo and cite our paper:
```
@article{zhang2023uformer,
  title={Uformer-ICS: A U-Shaped Transformer for Image Compressive Sensing Service},
  author={Zhang, Kuiyuan and Hua, Zhongyun and Li, Yuanman and Zhang, Yushu and Zhou, Yicong},
  journal={IEEE Transactions on Services Computing},
  year={2023},
  publisher={IEEE}
}
```
