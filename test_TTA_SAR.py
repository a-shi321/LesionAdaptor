# --------------------------------------------------------
# 修改后的Faster R-CNN测试代码，集成SAR算法
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
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ==================== SAR算法核心实现 ====================

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """计算softmax分布的熵"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_ema(ema, new_data):
    """更新指数移动平均"""
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def collect_params(model, skip_layers=['layer4', 'blocks.9', 'blocks.10', 'blocks.11']):
    """
    收集需要更新的参数（Norm层的scale和shift）
    跳过深层特征层以保持稳定性
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # 跳过深层特征层
        skip = False
        for skip_layer in skip_layers:
            if skip_layer in nm:
                skip = True
                break
        if skip:
            continue

        if 'norm.' in nm or nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def configure_model_for_sar(model):
    """
    配置模型用于SAR适应
    只启用Norm层的梯度
    """
    model.train()  # SAR需要训练模式
    model.requires_grad_(False)  # 默认禁用所有梯度

    # 启用Norm层的梯度
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # 强制使用batch统计量
            m.track_running_stats = False
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)

    return model


def copy_model_and_optimizer(model, optimizer):
    """复制模型和优化器状态用于重置"""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """恢复模型和优化器状态"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


class OptimizerWithSAR:
    """
    SAR优化器包装类
    实现Sharpness-Aware Minimization (SAM)的两步优化
    """

    def __init__(self, optimizer, rho=0.05):
        self.optimizer = optimizer
        self.rho = rho
        self.state = {}

    def zero_grad(self):
        self.optimizer.zero_grad()

    def first_step(self, zero_grad=False):
        """第一步：计算扰动"""
        grad_norm = self._grad_norm()
        if grad_norm == 0 or grad_norm is None:
            return

        rho_tensor = torch.tensor(self.rho, device=grad_norm.device)
        scale = rho_tensor / (grad_norm + 1e-12)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad.data * scale.to(p.device)
                p.data.add_(e_w)
                self.state[p] = e_w

        if zero_grad:
            self.optimizer.zero_grad()

    def second_step(self, zero_grad=False):
        """第二步：更新参数"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.state:
                    continue
                p.data.sub_(self.state[p])

        self.optimizer.step()
        self.state.clear()

        if zero_grad:
            self.optimizer.zero_grad()

    def _grad_norm(self):
        """计算梯度范数"""
        grad_norm = 0.0
        device = None
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.grad.device
                g = p.grad.detach()
                grad_norm += g.pow(2).sum()

        if device is None:
            return None

        grad_norm = torch.sqrt(grad_norm)

        if not torch.isfinite(grad_norm) or grad_norm.item() < 1e-12:
            return None

        return grad_norm

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def forward_and_adapt_sar_batch(x, model, sar_optimizer, margin_e0,
                                reset_constant_em, ema):
    """
    SAR的前向传播和适应步骤
    返回完整的12个输出值 + 统计信息
    """
    sar_optimizer.zero_grad()

    # 前向传播
    outputs = model(x[0], x[1], x[2], x[3])
    rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
    RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2 = outputs

    # 处理维度: [batch_size, num_rois, ...] -> [num_rois, ...]
    if cls_prob.dim() == 3:
        cls_prob_flat = cls_prob.squeeze(0)
    else:
        cls_prob_flat = cls_prob

    # 如果检测到的目标太少，跳过适应
    min_samples = 4
    if cls_prob_flat.size(0) < min_samples:
        return rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2, \
               ema, False

    # 计算熵
    entropys = softmax_entropy(cls_prob_flat)

    # 第一层过滤
    filter_ids_1 = torch.where(entropys < margin_e0)

    if filter_ids_1[0].size(0) == 0:
        return rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2, \
               ema, False

    entropys_filtered = entropys[filter_ids_1]
    loss = entropys_filtered.mean(0)

    # 第一步：计算扰动
    loss.backward()
    sar_optimizer.first_step(zero_grad=True)

    # 第二次前向传播
    outputs2 = model(x[0], x[1], x[2], x[3])
    _, _, cls_prob2, _, _, _, _, _, _, _, _, _ = outputs2

    if cls_prob2.dim() == 3:
        cls_prob2_flat = cls_prob2.squeeze(0)
    else:
        cls_prob2_flat = cls_prob2

    entropys2 = softmax_entropy(cls_prob2_flat)

    # 使用第一次 forward 得到的 reliable ROI 索引。
    # 正常情况下两次 Faster R-CNN forward 的 ROI 数量应一致；
    # 若由于实现细节发生变化，只保留合法索引，不改变 SAR 的筛选原则。
    reliable_idx = filter_ids_1[0]
    valid_idx = reliable_idx[reliable_idx < entropys2.size(0)]
    entropys2_filtered = entropys2[valid_idx]

    if entropys2_filtered.size(0) > 0:
        loss_second_value = entropys2_filtered.clone().detach().mean(0)
        if not math.isnan(loss_second_value.item()):
            ema = update_ema(ema, loss_second_value.item())

    if entropys2_filtered.size(0) > 0:
        filter_ids_2 = torch.where(entropys2_filtered < margin_e0)
        if filter_ids_2[0].size(0) > 0:
            loss_second = entropys2_filtered[filter_ids_2].mean(0)
            if not math.isnan(loss_second.item()):
                loss_second.backward()
                sar_optimizer.second_step(zero_grad=True)

    reset_flag = False
    if ema is not None:
        if ema < reset_constant_em:
            print(f"EMA ({ema:.4f}) < {reset_constant_em}, resetting model")
            reset_flag = True

    return rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
           RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2, \
           ema, reset_flag


class SARDetector(nn.Module):
    """SAR检测器包装类"""

    def __init__(self, model, sar_optimizer, steps=1, episodic=False,
                 margin_e0=0.4 * math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.sar_optimizer = sar_optimizer
        self.steps = steps
        self.episodic = episodic

        self.margin_e0 = margin_e0
        self.reset_constant_em = reset_constant_em
        self.ema = None

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.sar_optimizer.optimizer)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2, \
            ema, reset_flag = forward_and_adapt_sar_batch(
                (im_data, im_info, gt_boxes, num_boxes),
                self.model, self.sar_optimizer,
                self.margin_e0, self.reset_constant_em, self.ema
            )
            self.ema = ema

            if reset_flag:
                self.reset()
                rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2, \
                ema, _ = forward_and_adapt_sar_batch(
                    (im_data, im_info, gt_boxes, num_boxes),
                    self.model, self.sar_optimizer,
                    self.margin_e0, self.reset_constant_em, None
                )
                self.ema = ema

        return rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.sar_optimizer.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


# ==================== 主测试代码 ====================

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
    print("测试数据集是哪个？", args.imdbval_name)
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    # 初始化网络
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

    print("load checkpoint %s" % (args.load_name))
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    if args.cuda:
        fasterRCNN.cuda()

    # ============ SAR配置 ============
    USE_SAR = True
    SAR_STEPS = 1
    MARGIN_E0 = 0.4 * math.log(1000)
    RESET_CONSTANT_EM = 0.2
    SAR_LEARNING_RATE = 0.00025
    SAM_RHO = 0.05

    # 准备数据加载器
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    data_iter = iter(dataloader)

    if USE_SAR:
        fasterRCNN = configure_model_for_sar(fasterRCNN)

        params, param_names = collect_params(fasterRCNN, skip_layers=['layer4', 'blocks.9', 'blocks.10', 'blocks.11'])
        print(f"可训练参数数量: {len(params)}")
        for name in param_names:
            print(f"  - {name}")

        base_optimizer = torch.optim.SGD(params, SAR_LEARNING_RATE, momentum=0.9)
        sar_optimizer = OptimizerWithSAR(base_optimizer, rho=SAM_RHO)

        adapt_model = SARDetector(
            fasterRCNN, sar_optimizer,
            steps=SAR_STEPS,
            episodic=False,
            margin_e0=MARGIN_E0,
            reset_constant_em=RESET_CONSTANT_EM
        )
    else:
        adapt_model = fasterRCNN
        adapt_model.eval()

    # 初始化
    max_per_image = 200
    thresh = 0.0

    save_name = args.load_name.split('/')[-1]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)

    det_file = os.path.join(output_dir, 'detections.pkl')
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    time_start = time.time()

    for i in range(num_images):
        print(f"处理第 {i} 张测试图")
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data = next(data_iter)

        im_data = data[0].cuda() if args.cuda else data[0]
        im_info = data[1].cuda() if args.cuda else data[1]
        gt_boxes = data[2].cuda() if args.cuda else data[2]
        num_boxes = data[3].cuda() if args.cuda else data[3]

        det_tic = time.time()

        # ============ 使用SAR进行推理 ============
        if USE_SAR:
            rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2 = \
                adapt_model(im_data, im_info, gt_boxes, num_boxes)
        else:
            with torch.no_grad():
                rois, cls_score, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat, base_feat1, base_feat2 = \
                    adapt_model(im_data, im_info, gt_boxes, num_boxes)

        # ============ 后处理 ============
        # rois: [1, num_rois, 5]
        # 直接使用索引提取boxes，保持维度
        boxes = rois[:, :, 1:5]  # [1, num_rois, 4]

        # cls_prob: [1, num_rois, num_classes]
        scores = cls_prob  # [1, num_rois, num_classes]

        # bbox_pred: [1, num_rois, num_classes * 4]
        box_deltas = bbox_pred  # [1, num_rois, num_classes * 4]

        if cfg.TEST.BBOX_REG:
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            # boxes: [1, num_rois, 4], box_deltas: [1, num_rois, num_classes * 4]
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            # 去掉batch维度变为 [num_rois, 4]
            if pred_boxes.dim() == 3:
                pred_boxes = pred_boxes.squeeze(0)
        else:
            # 去掉batch维度
            if boxes.dim() == 3:
                boxes = boxes.squeeze(0)
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # 归一化坐标
        pred_boxes /= data[1][0][2].item()

        # 去掉batch维度
        if scores.dim() == 3:
            scores = scores.squeeze(0)
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes.squeeze(0)

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

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

                # 使用 .detach().cpu().numpy() 而不是 .cpu().numpy()
                all_boxes[j][i] = cls_dets.detach().cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # 限制每张图片的检测数量
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

        # 打印SAR统计信息
        if USE_SAR and i % 50 == 0:
            ema_value = adapt_model.ema if adapt_model.ema is not None else 0.0
            print(f"SAR - EMA: {ema_value:.4f}")

    end = time.time()
    each_time = (end - time_start) / num_images
    print("fps time: %0.4fs" % (each_time))
    print("fps: %0.2f" % (1.0 / each_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
