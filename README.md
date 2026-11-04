# LesionAdaptor

This project provides the implementation of the "LesionAdaptor: Dual-View Statistical Fusion and Reliability-Guided Optimization for Test-Time Adaptation in Lesion Detection", which achieve state-of-the-art performance on the online test-time domain adaptive lesion detection task.

## Preparation
Requirements: Python=3.6 and Pytorch=1.0.0

Install Pytorch

Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), please setup the framework by it.

## Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., [VGG16](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth) and [ResNet101](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth). Please download these two models.


## Train
Source domain train:
CUDA_VISIBLE_DEVICES=$GPU_ID python  trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset *** --dataset_t  ***   --save_dir ***


## Test
Source model test:

CUDA_VISIBLE_DEVICES=$GPU_ID test_SFso.py  --dataset *** --net res101 --cuda --load_name *********************.pth

and OTTAOD model test:

CUDA_VISIBLE_DEVICES=$GPU_ID  python test_TTA_ours.py --dataset  *** --net res101 --cuda --load_name *********************.pth
