# --------------------------------------------------------
# Faster R-CNN测试代码，集成COME (ICLR 2025) 算法
# 基于原有代码结构，只修改必要部分
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import time
import _init_paths
import cv2
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

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


# ==================== COME核心函数（新增） ====================

@torch.jit.script
def dirichlet_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet熵 - COME (ICLR 2025) 的核心创新
    通过引入"不确定性"类别来更好地区分模糊预测和确定预测
    """
    # 归一化logits
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()

    # 计算brief（确定性部分）和uncertainty（不确定性部分）
    sum_exp = torch.sum(torch.exp(x), dim=1, keepdim=True)
    brief = torch.exp(x) / (sum_exp + 1000)
    uncertainty = 1000 / (sum_exp + 1000)

    # 组合概率分布（增加一个"不确定性"类别）
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7

    # 计算熵
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy


# ==================== 原有函数（保持不变） ====================

def collect_params(model):
    print("开始搜集BN参数")
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names



def configure_model(model):
    """Configure model for use with tent_come."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model

# 在代码开头定义24种GBR颜色
COLORS_24_GBR = [
    (0, 255, 128),  # 亮橙绿
    (128, 255, 0),  # 亮黄绿
    (0, 128, 255),  # 亮橙红
    (235, 250, 230),  # 亮青绿
    (255, 128, 0),  # 亮青橙
    (0, 255, 200),  # 亮绿橙
    (200, 255, 0),  # 亮黄青
    (0, 200, 200),  # 亮橙黄
    (200, 200, 0),  # 亮青黄
]

def vis_detections(im, class_name, j, dets, thresh=0.8, colors=COLORS_24_GBR):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
                cv2.rectangle(im, bbox[0:2], bbox[2:4], colors[j], 3)#gbr
                retval, baseLine = cv2.getTextSize(class_name, fontFace=0, fontScale=1.0, thickness=3)
                # 计算覆盖文本的矩形框坐标
                topleft = (bbox[0], bbox[1] - retval[1] - 5)
                bottomright = (topleft[0] + retval[0], topleft[1] + retval[1] + 5)
                cv2.rectangle(im, (bbox[0], bbox[1] - 2 * baseLine), bottomright, colors[j], thickness=-1)
                cv2.putText(im,'%s' % (class_name), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), thickness=2)
    return im

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args, test=True)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    # initilize the network here.
    from model.TTA_faster_rcnn.resnet import resnet
    from model.TTA_faster_rcnn_vgg.vgg16 import vgg16

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    load_name = args.load_name
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})

    # ============ COME配置（新增） ============
    # 设置为True使用COME的Dirichlet熵，设置为False使用标准熵（Tent）
    USE_COME = True  # 改为False可对比Tent效果

    # ============ 配置模型（原有） ============
    fasterRCNN = configure_model(fasterRCNN)

    for name, value in fasterRCNN.named_parameters():
        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    params, _ = collect_params(fasterRCNN)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 200

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0


    save_name = args.load_name.split('/')[-1]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    fasterRCNN.train()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    time_start = time.time()

    # COME统计信息
    total_entropy = 0.0
    adapt_count = 0

    for i in range(num_images):
        print("第i张测试图", i)
        data = next(data_iter)
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, _, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, _, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        cls_prob = cls_prob.view(rois.size(1), -1)
        print("维度：", cls_prob.size())

        # ============ COME损失计算（修改部分） ============
        # 计算log概率
        log_prob = torch.log(cls_prob)

        # 使用COME的Dirichlet熵替代标准熵
        if USE_COME:
            # COME: 使用Dirichlet熵
            entropy = dirichlet_entropy(cls_prob)
            loss = entropy.mean()
        else:
            # Tent: 使用标准熵（原有方式）
            loss = -torch.sum(cls_prob * log_prob, dim=1).mean()

        # 反向传播（保持不变）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录统计信息
        if i % 10 == 0:
            print(f"  Loss: {loss.item():.4f}")

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], j, cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                         .format(i + 1, num_images, nms_time))
        sys.stdout.flush()
        if vis:
            # 创建保存目录（如果不存在）
            save_dir = os.path.join(output_dir, 'vis_results')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存图片，使用图片索引命名
            save_path = os.path.join(save_dir, f'{i:06d}_COME.png')
            cv2.imwrite(save_path, im2show)
            print(f"已保存可视化结果: {save_path}")
    end = time.time()
    each_time = (end - time_start) / num_images
    print("fps time: %0.4fs" % (each_time))
    print("fps: %0.2f" % (1.0 / each_time))

    # ============ 打印COME统计信息 ============
    print("\n" + "=" * 50)
    if USE_COME:
        print(">>> Using COME (Dirichlet Entropy) <<<")
    else:
        print(">>> Using Tent (Standard Entropy) <<<")
    print("=" * 50)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)