# MIST: Multiple Instance Spatial Transformer Network
### Baptiste Angles, Yuhe Jin, Simon Kornblith, Andrea Tagliasacchi, Kwang Moo Yi
This repository contains training and inference code for [MIST: Multiple Instance Spatial Transformer Network](https://arxiv.org/abs/1811.10725).

![alt text](https://github.com/jyh2005xx/mist-cleanup/blob/master/images/mist_pipeline.png)
## Installation
This code is implemented based on PyTorch. A conda environment is provided with all the dependencies:
```
conda env create -f system/conda_mist.yaml
```
## Pretrained models and datasets
Two pretrained models are provided for MNIST dataset and trimmed Pascal+COCO dataset respectively.
Models download path:
```
mkdir pretrained_models
wget https://www.cs.ubc.ca/research/kmyi_data/files/2021/mist/mnist_best_models -P ./pretrained_models/
wget https://www.cs.ubc.ca/research/kmyi_data/files/2021/mist/pascal_coco_best_models -P ./pretrained_models/
```
Dataset download path:
```
mkdir dataset
wget https://www.cs.ubc.ca/research/kmyi_data/files/2021/mist/mnist_hard.zip -P ./dataset/
wget https://www.cs.ubc.ca/research/kmyi_data/files/2021/mist/VOC_pascal_coco_v2.zip -P ./dataset/
unzip ./dataset/mnist_hard.zip -d ./dataset/
unzip ./dataset/VOC_pascal_coco_v2.zip -d ./dataset/
```
## Inference
Following commands will run pretrained model on test set. Visualization can be found in './test_results'
```
python mist_test.py --path_json='json/pascal.json'
python mist_test.py --path_json='json/mnist.json'
```
## Citation
```
@inproceedings{angles2021mist,
  title={MIST: Multiple Instance Spatial Transformer Networks},
  author={Baptiste Angles*, Yuhe Jin*, Simon Kornblith, Andrea Tagliasacchi, Kwang Moo Yi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
