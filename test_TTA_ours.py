
# ~~~~~~~~
# Results:
# 0.421
# 0.311
# 0.442
# 0.240
# 0.298
# 0.398
# 0.181
# 0.297
# 0.324
# ~~~~~~~~

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import time
import _init_paths

import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.parser_func import parse_args, set_dataset_args
import argparse
import pdb
import random
import torch.nn.functional as F


try:
    xrange
except NameError:
    xrange = range


# --------------------------------------------------------
# Random Seed
# --------------------------------------------------------

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

# --------------------------------------------------------
# Exp-3 Hyper Parameters
# --------------------------------------------------------

# --------------------------------------------------------
# Foreground threshold
#
# p(foreground) = 1 - p(background)
# --------------------------------------------------------

FOREGROUND_THRESHOLD = 0.50
# --------------------------------------------------------
# Teacher confidence threshold
#
# Only reliable AdaBN predictions participate
# in adaptation.
# --------------------------------------------------------

TEACHER_CONF_THRESHOLD = 0.50
# --------------------------------------------------------
# Teacher entropy threshold
#
# This is normalized according to number of classes.
#
# We use it only for selecting reliable instances.
# It is NOT an additional optimization loss.
# --------------------------------------------------------
TEACHER_ENTROPY_THRESHOLD = 0.80
# --------------------------------------------------------
# Student / teacher consistency threshold
# Only instances whose teacher and student predictions
# are not completely contradictory are used.
# This prevents highly unstable proposals from directly
# affecting BN parameters.
# --------------------------------------------------------
PREDICTION_AGREEMENT_THRESHOLD = 0.25
# --------------------------------------------------------
# Consistency loss weight
# IMPORTANT:
# This is FIXED.
# We are NOT introducing dynamic lambda.
# --------------------------------------------------------

CONSISTENCY_WEIGHT = 1.0
# --------------------------------------------------------
# Minimum selected instances
#
# If too few reliable instances exist, the adaptation
# will use the strongest teacher instances.
# --------------------------------------------------------

MIN_SELECTED_INSTANCES = 5
# --------------------------------------------------------
# Maximum selected instances
#
# Avoid allowing hundreds of noisy proposals to dominate
# the image-level BN update.
# --------------------------------------------------------

MAX_SELECTED_INSTANCES = 64
# --------------------------------------------------------
# Upper Branch
# AdaBN Reference Branch
# --------------------------------------------------------

def configure_adabn_reference(model, net):
    model.eval()
    # ----------------------------------------------------
    # Freeze entire model
    # ----------------------------------------------------
    for m in model.modules():
        for p in m.parameters():
            p.requires_grad = False
    # ----------------------------------------------------
    # Configure BN
    # ----------------------------------------------------
    if net == "res101":
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and ('RCNN_base1' in nm or'RCNN_base2' in nm):
                print("AdaBN reference BN:", nm)
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    elif net == "vgg16":
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print("AdaBN reference BN:", nm)
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    else:
        print("确定正确的网络结构")
    return model


# --------------------------------------------------------
# Lower Branch
# AdaMixBN Adaptive Branch
# --------------------------------------------------------

def configure_adamixbn(model, net):
    model.train()

    # ----------------------------------------------------
    # Freeze all parameters
    # ----------------------------------------------------
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    # ----------------------------------------------------
    # Enable BN affine parameters
    # ----------------------------------------------------
    if net == "res101":
        for nm, m in model.named_modules():

            if isinstance(m, nn.BatchNorm2d) and ('RCNN_base1' in nm or'RCNN_base2' in nm):
                print("AdaMixBN adaptive BN:", nm)
                m.train()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    elif net == "vgg16":
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print("AdaMixBN adaptive BN:", nm)
                m.train()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    else:
        print("确定正确的网络结构")
    return model


# --------------------------------------------------------
# Collect BN Parameters
# --------------------------------------------------------

def collect_params(model):
    params = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np_name, p in m.named_parameters():
                if np_name in ['weight', 'bias']:
                    if p.requires_grad:
                        params.append(p)
    return params


# --------------------------------------------------------
# Convert cls_score to [N, C]
#
# This function specifically fixes the previous problem:
#
# [1, 300, 9]
# [300, 9]
#
# are both accepted.
# --------------------------------------------------------

def normalize_cls_score(cls_score):
    if cls_score is None:
        raise ValueError("cls_score is None.")

    if cls_score.dim() == 3:
        if cls_score.size(0) != 1:
            raise ValueError(
                "Current implementation expects batch size 1, "
                "but got {}".format(cls_score.size())
            )
        cls_score = cls_score.squeeze(0)
    elif cls_score.dim() == 2:
        pass
    else:
        raise ValueError(
            "cls_score should be [N,C] or [1,N,C], "
            "but got {}".format(cls_score.size())
        )
    return cls_score


# --------------------------------------------------------
# Normalize cls_prob
# --------------------------------------------------------

def normalize_cls_prob(cls_prob):
    if cls_prob is None:
        raise ValueError("cls_prob is None.")
    if cls_prob.dim() == 3:
        if cls_prob.size(0) != 1:
            raise ValueError(
                "Current implementation expects batch size 1, "
                "but got {}".format(cls_prob.size())
            )
        cls_prob = cls_prob.squeeze(0)
    elif cls_prob.dim() == 2:
        pass
    else:
        raise ValueError(
            "cls_prob should be [N,C] or [1,N,C], "
            "but got {}".format(
                cls_prob.size() )
        )
    return cls_prob


# --------------------------------------------------------
# Entropy
# --------------------------------------------------------

def entropy_from_probability(prob):
    prob = prob.clamp(min=1e-6, max=1.0)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)

    return entropy


# --------------------------------------------------------
# Adaptive Instance Selection
# --------------------------------------------------------

def select_reliable_instances(cls_score_adabn, cls_score_adamixbn):
    # ----------------------------------------------------
    # Normalize dimensions
    # ----------------------------------------------------
    teacher_logits = normalize_cls_score(cls_score_adabn)
    student_logits = normalize_cls_score(cls_score_adamixbn)
    # ----------------------------------------------------
    # Make sure proposal numbers match
    # ----------------------------------------------------
    if teacher_logits.size(0) != student_logits.size(0):
        raise ValueError(
            "AdaBN and AdaMixBN proposal numbers "
            "do not match: {} vs {}".format(
                teacher_logits.size(),
                student_logits.size()
            )
        )
    # ----------------------------------------------------
    # Probability
    # ----------------------------------------------------
    teacher_prob = F.softmax(teacher_logits,dim=1)
    student_prob = F.softmax(student_logits,dim=1)
    # ----------------------------------------------------
    # Number of classes
    # ----------------------------------------------------
    num_classes = teacher_prob.size(1)

    if num_classes <= 1:
        raise ValueError("Number of classes must be > 1.")
    # ----------------------------------------------------
    # Background is class 0
    #
    # This follows Faster R-CNN convention in this code.
    # ----------------------------------------------------
    teacher_fg_prob = 1.0 - teacher_prob[:, 0]
    student_fg_prob = 1.0 - student_prob[:, 0]

    # ----------------------------------------------------
    # Teacher prediction
    # ----------------------------------------------------
    teacher_conf, teacher_label = torch.max(teacher_prob,dim=1)

    student_conf, student_label = torch.max(student_prob,dim=1)

    # ----------------------------------------------------
    # Teacher entropy
    #
    # Normalize by log(C)
    # ----------------------------------------------------

    teacher_entropy = entropy_from_probability(teacher_prob)

    normalized_teacher_entropy = (teacher_entropy / np.log(float(num_classes)))

    # ----------------------------------------------------
    # Student entropy
    # ----------------------------------------------------

    student_entropy = entropy_from_probability( student_prob)
    # ----------------------------------------------------
    # Basic foreground selection
    # ----------------------------------------------------

    foreground_mask = (teacher_fg_prob >FOREGROUND_THRESHOLD)
    # ----------------------------------------------------
    # Teacher confidence selection
    # ----------------------------------------------------

    confidence_mask = (teacher_conf > TEACHER_CONF_THRESHOLD)

    # ----------------------------------------------------
    # Teacher entropy selection
    # ----------------------------------------------------

    entropy_mask = (normalized_teacher_entropy <TEACHER_ENTROPY_THRESHOLD)

    # ----------------------------------------------------
    # Student foreground selection
    #
    # We don't require student confidence to be high.
    #
    # The purpose is to allow AdaMixBN to improve.
    # ----------------------------------------------------

    student_foreground_mask = (student_fg_prob >FOREGROUND_THRESHOLD * 0.5)

    # ----------------------------------------------------
    # Prediction agreement
    #
    # Instead of requiring identical labels,
    # compare probability assigned to teacher class.
    # ----------------------------------------------------

    teacher_class_prob_student = student_prob[
        torch.arange(student_prob.size(0), device=student_prob.device),
        teacher_label
    ]

    agreement_mask = (teacher_class_prob_student > PREDICTION_AGREEMENT_THRESHOLD)

    # ----------------------------------------------------
    # Final reliable mask
    # ----------------------------------------------------

    reliable_mask = (
        foreground_mask &
        confidence_mask &
        entropy_mask &
        student_foreground_mask &
        agreement_mask
    )

    # ----------------------------------------------------
    # Calculate quality score
    #
    # This score is ONLY used to rank proposals.
    #
    # It is NOT a dynamic lambda.
    # ----------------------------------------------------

    quality_score = (teacher_conf *teacher_fg_prob *(1.0 - normalized_teacher_entropy))

    # ----------------------------------------------------
    # Select reliable proposal indices
    # ----------------------------------------------------

    selected_indices = torch.nonzero(reliable_mask,as_tuple=False).view(-1)

    # ----------------------------------------------------
    # If too many instances exist,
    # keep only highest-quality instances.
    # ----------------------------------------------------

    if selected_indices.numel() > MAX_SELECTED_INSTANCES:

        selected_quality = quality_score[
            selected_indices
        ]

        _, order = torch.topk(
            selected_quality,
            k=MAX_SELECTED_INSTANCES,
            largest=True,
            sorted=False
        )

        selected_indices = selected_indices[
            order
        ]

    # ----------------------------------------------------
    # If too few reliable instances exist,
    # use top teacher-confidence foreground instances.
    #
    # This avoids completely disabling adaptation.
    # ----------------------------------------------------

    if selected_indices.numel() < MIN_SELECTED_INSTANCES:

        foreground_indices = torch.nonzero(
            foreground_mask,
            as_tuple=False
        ).view(-1)

        if foreground_indices.numel() > 0:

            fallback_k = min(
                MIN_SELECTED_INSTANCES,
                foreground_indices.numel()
            )

            fallback_quality = quality_score[
                foreground_indices
            ]

            _, order = torch.topk(
                fallback_quality,
                k=fallback_k,
                largest=True,
                sorted=False
            )

            selected_indices = foreground_indices[
                order
            ]

    return {
        "teacher_prob": teacher_prob,
        "student_prob": student_prob,
        "teacher_conf": teacher_conf,
        "student_conf": student_conf,
        "teacher_label": teacher_label,
        "student_label": student_label,
        "teacher_fg_prob": teacher_fg_prob,
        "student_fg_prob": student_fg_prob,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "normalized_teacher_entropy":
            normalized_teacher_entropy,
        "quality_score": quality_score,
        "foreground_mask": foreground_mask,
        "confidence_mask": confidence_mask,
        "entropy_mask": entropy_mask,
        "agreement_mask": agreement_mask,
        "reliable_mask": reliable_mask,
        "selected_indices": selected_indices
    }


# --------------------------------------------------------
# Instance Adaptation Loss
#
# Exp-3
#
# Only selected reliable instances participate.
# --------------------------------------------------------

def calculate_instance_adaptation_loss(cls_score_adabn, cls_score_adamixbn):
    # ----------------------------------------------------
    # Normalize
    # ----------------------------------------------------
    teacher_logits = normalize_cls_score(cls_score_adabn)

    student_logits = normalize_cls_score(cls_score_adamixbn)

    # ----------------------------------------------------
    # Select reliable instances
    # ----------------------------------------------------

    selection = select_reliable_instances(teacher_logits, student_logits)

    selected_indices = selection["selected_indices"]
    teacher_prob = selection["teacher_prob"]
    student_prob = selection["student_prob"]
    teacher_entropy = selection["teacher_entropy"]
    student_entropy = selection["student_entropy"]

    # ----------------------------------------------------
    # No selected instance
    # ----------------------------------------------------

    if selected_indices.numel() == 0:
        zero_loss = (student_logits.sum() * 0.0)
        return (zero_loss,zero_loss,zero_loss,selection)

    # ----------------------------------------------------
    # Selected predictions
    # ----------------------------------------------------
    selected_teacher_prob = teacher_prob[selected_indices]
    selected_student_prob = student_prob[selected_indices]
    selected_teacher_entropy = teacher_entropy[selected_indices]
    selected_student_entropy = student_entropy[selected_indices]

    # ----------------------------------------------------
    # 1. Instance entropy loss
    #
    # IMPORTANT:
    #
    # We ONLY minimize entropy for selected instances.
    # ----------------------------------------------------

    entropy_loss = (selected_student_entropy).mean()

    # ----------------------------------------------------
    # 2. Ranking loss
    #
    # AdaMixBN should not have higher entropy
    # than AdaBN on selected instances.
    #
    # margin = 0
    # ----------------------------------------------------
    ranking_loss = torch.clamp(selected_student_entropy - selected_teacher_entropy,min=0.0).mean()
    # ----------------------------------------------------
    # 3. Prediction consistency
    #
    # AdaBN acts as a detached teacher.
    #
    # KL(P_teacher || P_student)
    # ----------------------------------------------------

    teacher_target = (selected_teacher_prob.detach())
    teacher_log_prob = torch.log(teacher_target.clamp(min=1e-6))
    student_log_prob = torch.log(selected_student_prob.clamp(min=1e-6))
    consistency_loss = F.kl_div(student_log_prob,teacher_target, reduction='batchmean',log_target=False)

    # ----------------------------------------------------
    # Final loss
    #
    # No dynamic lambda.
    #
    # Fixed consistency weight = 1.0
    # ----------------------------------------------------

    final_loss = (entropy_loss + ranking_loss +CONSISTENCY_WEIGHT * consistency_loss)

    return (final_loss,  entropy_loss, ranking_loss,selection, consistency_loss)


# --------------------------------------------------------
# Detection Post Processing
# --------------------------------------------------------

def process_detections(rois,cls_prob,bbox_pred,im_info, data,imdb,all_boxes,i,args,thresh,max_per_image, empty_array):

    scores = cls_prob.detach()

    boxes = rois.detach()[:,:, 1:5]

    # ----------------------------------------------------
    # Bounding Box Regression
    # ----------------------------------------------------

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.detach()
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4)
                stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(box_deltas.device)
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(box_deltas.device)
                box_deltas = (box_deltas *stds + means)
                box_deltas = box_deltas.view( 1,  -1,   4)
            else:
                box_deltas = box_deltas.view(-1, 4 )
                stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(box_deltas.device)
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(box_deltas.device)
                box_deltas = (box_deltas * stds + means)
                box_deltas = box_deltas.view(1,-1, 4 * len(imdb.classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes,im_info.detach(), 1)

    else:

        pred_boxes = np.tile(scores.new_tensor(boxes.detach()).cpu().numpy(), ( 1,scores.shape[1]))

    # ----------------------------------------------------
    # Image Scale
    # ----------------------------------------------------

    pred_boxes /= data[1][0][2].item()
    scores = scores.squeeze(0)
    pred_boxes = pred_boxes.squeeze(0)
    # ----------------------------------------------------
    # Per-Class NMS
    # ----------------------------------------------------
    for j in xrange(1, imdb.num_classes):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)

        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0,True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, : ]
            else:
                cls_boxes = pred_boxes[inds][:,j * 4:(j + 1) * 4]
            cls_dets = torch.cat(( cls_boxes, cls_scores.unsqueeze(1)),1)
            cls_dets = cls_dets[ order]
            keep = nms( cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
            cls_dets = cls_dets[ keep.view(-1).long()]
            all_boxes[j][i] = ( cls_dets.detach() .cpu().numpy())

        else:
            all_boxes[j][i] = empty_array

    # ----------------------------------------------------
    # Maximum detections
    # ----------------------------------------------------

    if max_per_image > 0:
        valid_scores = []
        for j in xrange(1,imdb.num_classes):
            if ( all_boxes[j][i] is not None and
                    all_boxes[j][i].size > 0):
                valid_scores.append(
                    all_boxes[j][i][:, -1] )

        if len(valid_scores) > 0:
            image_scores = np.hstack( valid_scores)

            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange( 1,imdb.num_classes):
                    if (all_boxes[j][i] is not None and all_boxes[j][i].size > 0):
                        keep = np.where(all_boxes[j][i][:, -1]>= image_thresh)[0]
                        all_boxes[j][i] = (all_boxes[j][i][keep, :])
# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args,test=True)

    if (torch.cuda.is_available() and not args.cuda):
        print("WARNING: You have a CUDA device, " "so you should probably run with --cuda" )
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file( args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print( 'Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    # ----------------------------------------------------
    # Dataset
    # ----------------------------------------------------

    print( "测试数据集是哪个？", args.imdbval_name)

    imdb, roidb, ratio_list, ratio_index = (combined_roidb( args.imdbval_name, False))
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))

    load_name = args.load_name

    # ----------------------------------------------------
    # Networks
    # ----------------------------------------------------

    from model.TTA_faster_rcnn.resnet import resnet
    from model.TTA_faster_rcnn.resnet_fix import resnet_fix

    from model.TTA_faster_rcnn_vgg.vgg16 import vgg16
    from model.TTA_faster_rcnn_vgg.vgg16_fix import vgg16_fix

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes,pretrained=False, class_agnostic=args.class_agnostic)
        fasterRCNN_fix = vgg16_fix(imdb.classes,pretrained=False,class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes,101,pretrained=True,class_agnostic=args.class_agnostic)
        fasterRCNN_fix = resnet_fix(imdb.classes,101,pretrained=True,class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50,  pretrained=True,class_agnostic=args.class_agnostic)
        fasterRCNN_fix = resnet_fix( imdb.classes,50,pretrained=True,class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

        pdb.set_trace()

    # ----------------------------------------------------
    # Create Architecture
    # ----------------------------------------------------

    fasterRCNN.create_architecture()
    fasterRCNN_fix.create_architecture()

    # ----------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------

    print("load checkpoint %s" % load_name)

    checkpoint = torch.load(load_name)

    fasterRCNN.load_state_dict({ k: v  for k, v in checkpoint['model'].items() if k in fasterRCNN.state_dict()}, False)
    fasterRCNN_fix.load_state_dict({ k: v  for k, v in checkpoint['model'].items() if k in fasterRCNN_fix.state_dict()}, False)

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    # ----------------------------------------------------
    # Configure AdaBN
    # ----------------------------------------------------

    print("")
    print("==============================================")

    print("Configure Upper Branch : AdaBN Reference")

    print("==============================================")

    fasterRCNN_fix = configure_adabn_reference(fasterRCNN_fix, args.net)

    # ----------------------------------------------------
    # Configure AdaMixBN
    # ----------------------------------------------------
    print("")
    print("==============================================")
    print("Configure Lower Branch : AdaMixBN Adaptive")
    print("==============================================")

    fasterRCNN = configure_adamixbn(fasterRCNN,args.net)

    # ----------------------------------------------------
    # Parameters
    # ----------------------------------------------------

    params = collect_params(fasterRCNN)

    print("Number of trainable AdaMixBN parameters:",len(params))

    # ----------------------------------------------------
    # Optimizer
    # ----------------------------------------------------

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params,lr=cfg.TRAIN.LEARNING_RATE)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params,lr=cfg.TRAIN.LEARNING_RATE,momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        print("optimizer is not defined")
        pdb.set_trace()

    # ----------------------------------------------------
    # CUDA
    # ----------------------------------------------------
    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
        fasterRCNN_fix.cuda()

    # ----------------------------------------------------
    # Tensor holder
    # ----------------------------------------------------

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    # ----------------------------------------------------
    # Detection configuration
    # ----------------------------------------------------

    max_per_image = 200

    thresh = 0.0

    save_name = load_name.split('/')[-1]

    num_images = len(imdb.image_index)

    all_boxes = [[[] for _ in xrange(num_images)]
        for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb,ratio_list,ratio_index, 1, imdb.num_classes,training=False,normalize=False)
    dataloader = torch.utils.data.DataLoader( dataset,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    data_iter = iter(dataloader)
    det_file = os.path.join(output_dir,'detections.pkl')
    empty_array = np.transpose(np.array([ [],[],[],[],[]]),(1, 0))

    # ----------------------------------------------------
    # Make sure BN is using target statistics
    # ----------------------------------------------------

    fasterRCNN.train()
    fasterRCNN_fix.eval()

    for m in fasterRCNN_fix.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    valid_images = 0
    foreground_instances = 0
    selected_instances = 0
    entropy_better_count = 0
    entropy_worse_count = 0
    entropy_equal_count = 0
    total_teacher_entropy = 0.0
    total_student_entropy = 0.0
    total_entropy_advantage = 0.0
    total_consistency_loss = 0.0
    total_ranking_loss = 0.0
    total_entropy_loss = 0.0
    total_prediction_drift = 0
    total_selected_drift = 0

    # ----------------------------------------------------
    # Time
    # ----------------------------------------------------

    time_start = time.time()

    # ----------------------------------------------------
    # Main loop
    # ----------------------------------------------------

    for i in range(num_images):

        print("")
        print("==============================================")
        print("Instance-Level Exp-3 adaptive testing " "{}/{}".format( i,num_images))
        print( "==============================================")
        try:
            data = next(data_iter)
        except StopIteration:
            break
        # ------------------------------------------------
        # Load image
        # ------------------------------------------------
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])

        # ------------------------------------------------
        # No GT during TTA
        # ------------------------------------------------
        gt_boxes.resize_(data[2].size()).zero_()
        num_boxes.resize_(data[3].size()).zero_()

        # ------------------------------------------------
        # Clear gradients
        # ------------------------------------------------
        optimizer.zero_grad()
        fasterRCNN.zero_grad()
        fasterRCNN_fix.zero_grad()

        # ------------------------------------------------
        # Stage 1
        #
        # AdaBN teacher
        # ------------------------------------------------

        with torch.no_grad():
            (rois_fix,cls_score_fix,cls_prob_fix,bbox_pred_fix,rpn_loss_cls_fix,rpn_loss_bbox_fix,RCNN_loss_cls_fix,
             RCNN_loss_bbox_fix,rois_label_fix,base_feat_fix,base_feat1_fix,base_feat2_fix) = fasterRCNN_fix(im_data,im_info,gt_boxes,num_boxes)
        # ------------------------------------------------
        # Stage 2
        #
        # AdaMixBN student
        # ------------------------------------------------

        (rois,cls_score,cls_prob, bbox_pred,rpn_loss_cls,rpn_loss_bbox,RCNN_loss_cls,RCNN_loss_bbox,
          rois_label,base_feat,base_feat1,base_feat2) = fasterRCNN(im_data,im_info,gt_boxes,num_boxes)
        # ------------------------------------------------
        # Debug shape
        # ------------------------------------------------

        print( "AdaBN cls_score shape:", cls_score_fix.shape)

        print("AdaMixBN cls_score shape:",cls_score.shape)

        # ------------------------------------------------
        # Normalize scores
        # ------------------------------------------------
        teacher_logits = normalize_cls_score(cls_score_fix)
        student_logits = normalize_cls_score(cls_score)
        # ------------------------------------------------
        # Calculate selection and loss
        # ------------------------------------------------
        (loss_final, entropy_loss, ranking_loss, selection,consistency_loss) = calculate_instance_adaptation_loss( teacher_logits, student_logits)
        # ------------------------------------------------
        # Statistics
        # ------------------------------------------------

        teacher_entropy_all = entropy_from_probability(F.softmax(teacher_logits,dim=1))
        student_entropy_all = entropy_from_probability(F.softmax(student_logits,dim=1))

        teacher_prob_all = F.softmax(teacher_logits,dim=1)
        student_prob_all = F.softmax(student_logits,dim=1)
        teacher_fg_all = (1.0 -teacher_prob_all[:, 0])
        foreground_mask = ( teacher_fg_all >FOREGROUND_THRESHOLD)

        valid_fg_teacher_entropy = (teacher_entropy_all[foreground_mask])
        valid_fg_student_entropy = (student_entropy_all[foreground_mask])

        num_fg = int(foreground_mask.sum().item())
        num_selected = int(selection[ "selected_indices"].numel())

        foreground_instances += num_fg
        selected_instances += num_selected

        # ------------------------------------------------
        # Entropy comparison
        #
        # Only foreground instances
        # ------------------------------------------------

        if num_fg > 0:
            valid_images += 1
            total_teacher_entropy += (valid_fg_teacher_entropy.mean().detach().item())
            total_student_entropy += (valid_fg_student_entropy.mean().detach().item())

            advantage = ( valid_fg_teacher_entropy -valid_fg_student_entropy)
            total_entropy_advantage += (advantage.mean().detach().item())
            entropy_better_count += int((advantage > 1e-6).sum().item())

            entropy_worse_count += int((advantage < -1e-6).sum().item())
            entropy_equal_count += int((advantage.abs() <= 1e-6).sum().item())

        # ------------------------------------------------
        # Prediction drift
        # ------------------------------------------------

        teacher_label_all = torch.argmax(teacher_prob_all,dim=1)
        student_label_all = torch.argmax(student_prob_all,dim=1)

        drift_mask = (foreground_mask & (teacher_label_all != student_label_all))
        total_prediction_drift += int( drift_mask.sum().item())

        selected_indices = selection["selected_indices"]

        if selected_indices.numel() > 0:
            selected_teacher_label = (teacher_label_all[selected_indices])
            selected_student_label = (student_label_all[selected_indices])
            selected_drift = (selected_teacher_label != selected_student_label)
            total_selected_drift += int(selected_drift.sum().item())

        total_consistency_loss += (consistency_loss.detach().item())
        total_ranking_loss += (ranking_loss.detach().item())
        total_entropy_loss += (entropy_loss.detach().item())

        # ------------------------------------------------
        # Backward
        # ------------------------------------------------

        if loss_final.requires_grad:
            loss_final.backward()
            optimizer.step()

        # ------------------------------------------------
        # Print adaptation information
        # ------------------------------------------------

        print("Foreground instances : {}".format(num_fg))
        print("Selected instances   : {}".format(num_selected))
        print("Teacher entropy      : {:.6f}".format(teacher_entropy_all.mean().detach().item() ))
        print("Student entropy      : {:.6f}".format(student_entropy_all.mean().detach().item()))
        print("Entropy loss         : {:.6f}".format(entropy_loss.detach().item()))
        print("Ranking loss         : {:.6f}".format( ranking_loss.detach().item()))
        print("Consistency loss     : {:.6f}" .format( consistency_loss.detach().item()))
        print( "Final loss           : {:.6f}" .format( loss_final.detach().item()))
        print( "Prediction drift     : {}".format(int(drift_mask.sum().item())))
        print( "Selected drift       : {}".format(total_selected_drift))

        # ------------------------------------------------
        # Stage 3
        #
        # Re-forward after adaptation
        # ------------------------------------------------
        with torch.no_grad():
            (rois_final, cls_score_final,cls_prob_final,bbox_pred_final,_,_,_,_,_,_, _,_) = fasterRCNN(im_data,im_info,gt_boxes,num_boxes)

        # ------------------------------------------------
        # Final detection
        # ------------------------------------------------
        process_detections(rois_final,cls_prob_final,bbox_pred_final,im_info, data,imdb, all_boxes,i,args,thresh,max_per_image,empty_array)
        # ------------------------------------------------
        # Progress
        # ------------------------------------------------
        sys.stdout.write('im_detect: {:d}/{:d}   \r'.format(i + 1, num_images))
        sys.stdout.flush()

    # ----------------------------------------------------
    # End
    # ----------------------------------------------------

    end = time.time()
    each_time = (end - time_start) / max(num_images,1)

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    total_instances = (entropy_better_count +entropy_worse_count + entropy_equal_count)

    print("")
    print("==================================================")
    print("Exp-3 Instance-Level Adaptive Selection Statistics")
    print( "==================================================")
    print("Valid images               : {}".format(valid_images))
    print("Foreground instances      : {}".format(foreground_instances))
    print("Selected instances        : {}".format(selected_instances))
    if foreground_instances > 0:
        print("Selection ratio            : {:.2f}%".format(100.0 *selected_instances /  float(foreground_instances) ))
    print("AdaMixBN entropy better   : {}" .format( entropy_better_count))
    print("AdaBN entropy better      : {}".format(entropy_worse_count))
    print("Entropy equal             : {}".format(entropy_equal_count))

    if total_instances > 0:
        print("AdaMixBN better ratio     : {:.2f}%".format(100.0 *entropy_better_count /float(total_instances)))
        print("AdaBN better ratio        : {:.2f}%".format(100.0 *entropy_worse_count /float(total_instances)))
        print("Equal ratio               : {:.2f}%".format(100.0 *entropy_equal_count /float(total_instances)))

    if valid_images > 0:

        avg_teacher_entropy = ( total_teacher_entropy / float(valid_images))
        avg_student_entropy = ( total_student_entropy / float(valid_images))
        avg_advantage = (total_entropy_advantage /float(valid_images))

        print("Average AdaBN entropy     : {:.6f}".format(avg_teacher_entropy))
        print("Average AdaMixBN entropy  : {:.6f}".format(avg_student_entropy))
        print("Average entropy advantage: {:.6f}".format(avg_advantage))
    print("Prediction drift          : {}".format(total_prediction_drift))
    print("Selected prediction drift : {}".format(total_selected_drift))

    if selected_instances > 0:
        print("Selected drift ratio      : {:.2f}%".format( 100.0 *total_selected_drift /float(selected_instances)))

    if num_images > 0:
        print("Average consistency loss  : {:.6f}".format( total_consistency_loss /float(num_images)))
        print("Average ranking loss      : {:.6f}".format(total_ranking_loss /float(num_images)))
        print("Average entropy loss      : {:.6f}".format(total_entropy_loss /float(num_images)))

    print("==================================================")
    print("test time: %0.4fs"% ( end -time_start))
    print("fps time: %0.4fs"% each_time)
    print("fps: %0.2f" % (1.0 /each_time))
    # ----------------------------------------------------
    # Save detection
    # ----------------------------------------------------
    with open( det_file,'wb') as f:
        pickle.dump(all_boxes,f,pickle.HIGHEST_PROTOCOL)
    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)