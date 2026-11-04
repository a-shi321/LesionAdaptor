# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from lib.model.SF_faster_rcnn_vgg.faster_rcnn import _fasterRCNN
import copy
from model.utils.config import cfg
import pdb
import os.path as osp



class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = osp.abspath(osp.join(cfg.DATA_DIR, 'pretrained_model/vgg16_bn.pth'))
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16_bn()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # # not using the last maxpool layer
        # self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        #
        # # Fix the layers before conv3:
        # for layer in range(14):
        #     for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.block1_1 = nn.Sequential(*list(vgg.features._modules.values())[0:2])  # conv-bn
        # self.block1_2 = nn.Sequential(*list(vgg.features._modules.values())[2:5])  # relu-conv-bn
        #
        # self.block1_3 = nn.Sequential(*list(vgg.features._modules.values())[5:7])  # relu-maxpooling
        #
        # self.block2_1 = nn.Sequential(*list(vgg.features._modules.values())[7:9])  # conv-bn
        # self.block2_2 = nn.Sequential(*list(vgg.features._modules.values())[9:12])  # relu-conv-bn
        #
        # self.block2_3 = nn.Sequential(*list(vgg.features._modules.values())[12:14])  # relu-maxpooling
        #
        # self.block3_1 = nn.Sequential(*list(vgg.features._modules.values())[14: 16])  # conv-bn
        # self.block3_2 = nn.Sequential(*list(vgg.features._modules.values())[16: 19])  # relu-conv-bn
        # self.block3_3 = nn.Sequential(*list(vgg.features._modules.values())[19: 22])  # relu-conv-bn
        #
        # self.block3_4 = nn.Sequential(*list(vgg.features._modules.values())[22: 24])  ##relu-maxpooling
        #
        # self.block4_1 = nn.Sequential(*list(vgg.features._modules.values())[24: 26])  # conv-bn
        # self.block4_2 = nn.Sequential(*list(vgg.features._modules.values())[26: 29])  # relu-conv-bn
        # self.block4_3 = nn.Sequential(*list(vgg.features._modules.values())[29: 32])  # relu-conv-bn
        #
        # self.block4_4 = nn.Sequential(*list(vgg.features._modules.values())[32: 34])  ##relu-maxpooling
        #
        # self.block5_1 = nn.Sequential(*list(vgg.features._modules.values())[34: 36])  # conv-bn
        # self.block5_2 = nn.Sequential(*list(vgg.features._modules.values())[36: 39])  # relu-conv-bn
        # self.block5_3 = nn.Sequential(*list(vgg.features._modules.values())[39: 42])  # relu-conv-bn
        # self.block5_4 = nn.Sequential(*list(vgg.features._modules.values())[42: -1])  # relu

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

