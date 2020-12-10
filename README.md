## Overview
This project uses the MaskRCNN network to slove the semantic segmentation challenge of Tiny Poscal dataset. The model is base on pytorch and torchvision library and reference to this [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1) 

## Installation
All requirements should be detailed in requirements.txt.
```
# python version: Python 3.6.9
pip3 install -r requirements.txt
```

## Dataset Preparation
1. download the training data from here [google drive](https://drive.ggle.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)
2. To training this model on our own dataset, we must follow the format defined [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset) and it was done here, `dataloader.py`.
3. By default, I set "./dataset" as the root directory of our dataset. You can set a symbolic link to the real path or pass the real path as a parameter.

## Training
- Using the following script to get more information
```
$ python train.py --help
```
- Example
```
python3 train.py --lr 0.001 --dataset "./dataset"
```
    
## Testing
- Add parameter, `--test` to forward the test data and `--save_json` to generate the prediction file with coco style format.
```
$ python train.py --weight <PATH_TO_TRAINED_MODEL_WEIGHT> --test --save_json
```
- The result would save under the `submissions` folder.

## Reference
- This project refers to this tutorial [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) page
- And some useful tools mentioned in the tutorial can be found [here](https://github.com/pytorch/vision/tree/master/references/detection)