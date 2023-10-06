# Uformer-ICS


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
    'set1024': "path_of/Set1024", # used for testing time complexity
    'test_for_visual_comparison':"path_of/test_for_visual_comparison" # used for visual comparison,
}
```

## Training and Testing


### Training scripts 

```bash
# train Uformer-ICS
python train.py --gpu 0 --cfg "Ours/coco/1";\
python train.py --gpu 0 --cfg "Ours/coco/4";\
python train.py --gpu 0 --cfg "Ours/coco/10";\
python train.py --gpu 0 --cfg "Ours/coco/25";\
python train.py --gpu 0 --cfg "Ours/coco/50";\

# train Uformer-ICS+
python train.py --gpu 0 --cfg "Ours/coco/999";
```

### Testing scripts

To test our methods, just add some arguments at the end of training scripts.

1. test on different datasets, e.g. Set5, Set11, Set14, BSD100...

```bash
# test Uformer-ICS
python train.py --gpu 0 --cfg "Ours/coco/1" --test 1;\

# test Uformer-ICS+
python train.py --gpu 0 --cfg "Ours/coco/999" --test 1;
```




