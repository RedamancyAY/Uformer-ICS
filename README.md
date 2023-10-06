# Uformer-ICS


## Prepare datasets

We use the COCO datasets for training.



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




