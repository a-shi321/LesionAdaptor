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
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
#from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func import parse_args,set_dataset_args
import argparse
import pdb
# 导入PIL库
from PIL import Image, ImageDraw, ImageFont
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3
def compute_confusion_matrix_no_background(all_boxes, imdb, gt_roidb, num_classes, iou_thresh=0.5):
    """
    计算混淆矩阵（只包含前景类别，不包含background）
    Args:
        all_boxes: 预测的检测框列表
        imdb: 数据集对象
        gt_roidb: ground truth roidb
        num_classes: 类别数量
        iou_thresh: IoU阈值
    Returns:
        conf_matrix: 混淆矩阵 (前景类别数 x 前景类别数)
        class_names: 前景类别名称列表
    """
    # 获取前景类别数量（去掉background）
    n_foreground = num_classes - 1
    class_names = list(imdb.classes)[1:] if isinstance(imdb.classes, (list, tuple)) else list(imdb.classes)[1:]

    # 初始化混淆矩阵
    conf_matrix = np.zeros((n_foreground, n_foreground), dtype=np.int64)

    # 统计每个图像的真实标签和预测标签
    for img_idx in range(len(imdb.image_index)):
        # 获取真实标签
        gt_boxes = gt_roidb[img_idx]['boxes']
        gt_classes = gt_roidb[img_idx]['gt_classes']
        gt_difficult = gt_roidb[img_idx]['gt_difficult'] if 'gt_difficult' in gt_roidb[img_idx] else np.zeros(
            len(gt_classes))

        # 获取预测结果（只包含前景类别）
        pred_boxes = []
        pred_scores = []
        pred_classes = []
        for class_idx in range(1, num_classes):  # 跳过背景类
            if len(all_boxes[class_idx][img_idx]) > 0:
                for det in all_boxes[class_idx][img_idx]:
                    pred_boxes.append(det[:4])
                    pred_scores.append(det[4])
                    pred_classes.append(class_idx - 1)  # 转换为0-based索引（去掉背景）

        if len(pred_boxes) == 0:
            # 如果没有预测，所有真实目标被标记为漏检（但这里我们不统计漏检到background）
            continue

        pred_boxes = np.array(pred_boxes)
        pred_classes = np.array(pred_classes)
        pred_scores = np.array(pred_scores)

        # 对每个真实目标，找到匹配的预测
        matched_pred = set()

        # 对每个真实目标找最佳匹配
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_class = gt_classes[gt_idx]
            if gt_class == 0:  # 跳过背景
                continue
            if len(gt_difficult) > gt_idx and gt_difficult[gt_idx] == 1:  # 忽略困难样本
                continue

            # 计算与所有预测框的IoU
            overlaps = np.zeros(len(pred_boxes))
            for pred_idx, pred_box in enumerate(pred_boxes):
                # 计算IoU
                x1 = max(gt_box[0], pred_box[0])
                y1 = max(gt_box[1], pred_box[1])
                x2 = min(gt_box[2], pred_box[2])
                y2 = min(gt_box[3], pred_box[3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    union = gt_area + pred_area - intersection
                    overlaps[pred_idx] = intersection / union
                else:
                    overlaps[pred_idx] = 0

            # 找到最佳匹配
            if len(overlaps) > 0:
                best_pred_idx = np.argmax(overlaps)
                best_iou = overlaps[best_pred_idx]

                if best_iou >= iou_thresh and best_pred_idx not in matched_pred:
                    # 匹配成功
                    pred_class = pred_classes[best_pred_idx]
                    conf_matrix[gt_class - 1, pred_class] += 1  # gt_class是1-based，转换为0-based
                    matched_pred.add(best_pred_idx)
                # 如果匹配失败（漏检），不统计到混淆矩阵中

    return conf_matrix, class_names


def plot_confusion_matrix_pil_foreground(conf_matrix, class_names, output_dir, normalize=True):
    """
    使用PIL绘制混淆矩阵图片（只包含前景类别，表格风格）
    Args:
        conf_matrix: 混淆矩阵 (前景类别数 x 前景类别数)
        class_names: 前景类别名称列表
        output_dir: 输出目录
        normalize: 是否归一化
    """
    # 将class_names转换为列表
    if isinstance(class_names, tuple):
        class_names = list(class_names)
    elif not isinstance(class_names, list):
        class_names = list(class_names)

    # 复制混淆矩阵
    conf_matrix_copy = conf_matrix.copy().astype(np.float64)

    if normalize:
        # 按行归一化
        row_sums = conf_matrix_copy.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        conf_matrix_display = conf_matrix_copy / row_sums
        suffix = '_normalized'
        title = 'Confusion Matrix (Normalized)'
    else:
        conf_matrix_display = conf_matrix_copy
        suffix = ''
        title = 'Confusion Matrix'

    n_classes = conf_matrix.shape[0]

    # 设置图片参数 - 调整大小适应更多类别
    if n_classes <= 4:
        cell_size = 120
        font_size = 14
    elif n_classes <= 6:
        cell_size = 100
        font_size = 12
    else:
        cell_size = 80
        font_size = 10

    padding = 30
    label_width = 100

    # 计算图片大小
    img_width = n_classes * cell_size + padding * 2 + label_width + 40
    img_height = n_classes * cell_size + padding * 2 + 100 + 40

    # 创建图片（使用白色背景）
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font_small = ImageFont.truetype("arial.ttf", max(font_size - 2, 8))
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_label = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    # 绘制标题
    draw.text((img_width // 2 - 100, 10), title, fill='black', font=font_title)

    # 颜色映射 - 使用更鲜艳的蓝色渐变
    colors = [
        (255, 255, 255),  # 白色
        (230, 240, 255),
        (200, 220, 255),
        (160, 190, 255),
        (120, 160, 255),
        (80, 130, 255),
        (40, 100, 255),
        (0, 70, 255),
        (0, 40, 200),
        (0, 0, 150),
    ]

    # 绘制每个格子
    for i in range(n_classes):
        for j in range(n_classes):
            x = padding + j * cell_size + label_width + 20
            y = padding + i * cell_size + 50

            # 获取数值
            value = conf_matrix_display[i, j]

            # 计算颜色索引
            if normalize:
                color_idx = min(int(value * 9), 8)
            else:
                # 对于原始数值，使用对数缩放
                if value > 0:
                    color_idx = min(int(np.log1p(value) * 2), 8)
                else:
                    color_idx = 0

            # 选择颜色
            if color_idx == 0 and value == 0:
                fill_color = (245, 245, 245)  # 浅灰色
            else:
                fill_color = colors[color_idx + 1] if color_idx + 1 < len(colors) else colors[-1]

            # 绘制矩形
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=fill_color, outline='#888888', width=1)

            # 绘制数值
            if normalize:
                text = f"{value:.2f}"
            else:
                text = f"{int(value)}"

            # 根据背景色选择文字颜色
            brightness = (fill_color[0] + fill_color[1] + fill_color[2]) / 3
            text_color = 'black' if brightness > 150 else 'white'

            # 获取文本大小并居中
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)

            text_x = x + (cell_size - text_width) // 2
            text_y = y + (cell_size - text_height) // 2
            draw.text((text_x, text_y), text, fill=text_color, font=font)

    # 绘制行标签（真实类别）- 靠右对齐
    for i in range(n_classes):
        y = padding + i * cell_size + 50 + cell_size // 2
        label = class_names[i]

        # 获取文本大小
        try:
            bbox = draw.textbbox((0, 0), label, font=font_label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label, font=font_label)

        # 绘制文本
        x = padding + label_width - 10
        draw.text((x - text_width, y - text_height // 2), label, fill='black', font=font_label)

    # 绘制列标签（预测类别）- 居中对齐
    for j in range(n_classes):
        x = padding + j * cell_size + label_width + 20 + cell_size // 2
        label = class_names[j]

        # 获取文本大小
        try:
            bbox = draw.textbbox((0, 0), label, font=font_label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label, font=font_label)

        # 如果标签太长，旋转显示
        if text_width > cell_size:
            label_img = Image.new('RGBA', (text_height + 10, text_width + 10), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((5, 5), label, fill='black', font=font_label)
            label_img = label_img.rotate(45, expand=True)

            paste_x = x - label_img.width // 2
            paste_y = padding + n_classes * cell_size + 60
            img.paste(label_img, (paste_x, paste_y), label_img)
        else:
            y = padding + n_classes * cell_size + 60
            draw.text((x - text_width // 2, y), label, fill='black', font=font_label)

    # 绘制坐标轴标签
    # X轴
    axis_y = padding + n_classes * cell_size + 85
    draw.text((padding + label_width + 20 + n_classes * cell_size // 2 - 50, axis_y),
              'Predicted Label', fill='#333333', font=font)

    # Y轴 - 垂直文本
    true_label_img = Image.new('RGBA', (100, 30), (255, 255, 255, 0))
    true_label_draw = ImageDraw.Draw(true_label_img)
    true_label_draw.text((0, 0), 'True Label', fill='#333333', font=font)
    true_label_img = true_label_img.rotate(90, expand=True)
    img.paste(true_label_img, (10, padding + n_classes * cell_size // 2 + 50), true_label_img)

    # 保存图片
    save_path = os.path.join(output_dir, f'confusion_matrix_foreground{suffix}.png')
    img.save(save_path, 'PNG', quality=95)
    print(f'混淆矩阵图片已保存: {save_path}')


def save_confusion_matrix_foreground(conf_matrix, class_names, output_dir):
    """
    保存混淆矩阵为文本文件（只包含前景类别）
    """
    # 将class_names转换为列表
    if isinstance(class_names, tuple):
        class_names = list(class_names)
    elif not isinstance(class_names, list):
        class_names = list(class_names)

    n_classes = conf_matrix.shape[0]

    # 保存混淆矩阵为文本文件
    txt_path = os.path.join(output_dir, 'confusion_matrix_foreground.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX (Foreground Classes Only)\n")
        f.write("=" * 80 + "\n\n")

        # 表头
        f.write("True \\ Pred\n")
        f.write(" " * 15)
        for name in class_names:
            f.write(f"{name:>12}")
        f.write("\n")
        f.write("-" * (15 + 12 * len(class_names)) + "\n")

        # 矩阵内容
        for i, name in enumerate(class_names):
            f.write(f"{name:>12}: ")
            for j in range(conf_matrix.shape[1]):
                f.write(f"{conf_matrix[i, j]:>12}")
            f.write("\n")

        # 计算评估指标
        f.write("\n" + "=" * 80 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Per-class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':>15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}\n")
        f.write("-" * 80 + "\n")

        precisions = []
        recalls = []
        f1s = []

        for i in range(n_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            support = conf_matrix[i, :].sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            f.write(f"{class_names[i]:>15} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {support:>12d}\n")

        # 整体指标
        f.write("\n" + "-" * 80 + "\n")
        f.write("Overall Metrics:\n")
        f.write("-" * 80 + "\n")

        total_correct = np.trace(conf_matrix)
        total_samples = np.sum(conf_matrix)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        f.write(f"{'Accuracy':>15} {accuracy:>12.4f}\n")

        macro_precision = np.mean(precisions) if precisions else 0
        macro_recall = np.mean(recalls) if recalls else 0
        macro_f1 = np.mean(f1s) if f1s else 0

        f.write(f"{'Macro Precision':>15} {macro_precision:>12.4f}\n")
        f.write(f"{'Macro Recall':>15} {macro_recall:>12.4f}\n")
        f.write(f"{'Macro F1-Score':>15} {macro_f1:>12.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f'混淆矩阵文本已保存: {txt_path}')

    # 同时打印到控制台
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Foreground Classes Only)")
    print("=" * 80)
    print("\nTrue \\ Pred")
    print(" " * 15, end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print("\n" + "-" * (15 + 12 * len(class_names)))
    for i, name in enumerate(class_names):
        print(f"{name:>12}: ", end="")
        for j in range(conf_matrix.shape[1]):
            print(f"{conf_matrix[i, j]:>12}", end="")
        print()

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"\n{'Class':>15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 80)
    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_names[i]:>15} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {support:>12d}")

    print("\n" + "-" * 80)
    print("Overall Metrics:")
    print("-" * 80)
    total_correct = np.trace(conf_matrix)
    total_samples = np.sum(conf_matrix)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"{'Accuracy':>15} {accuracy:>12.4f}")
    print(f"{'Macro Precision':>15} {macro_precision:>12.4f}")
    print(f"{'Macro Recall':>15} {macro_recall:>12.4f}")
    print(f"{'Macro F1-Score':>15} {macro_f1:>12.4f}")
    print("=" * 80 + "\n")

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
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    print("开始设计bn")
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print("输出！！！")
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
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
    # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)

    if args.part == 'test_s':
        print("输出目标域检测数据集名称", args.s_imdbtest_name)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.s_imdbtest_name, False)
    elif args.part == 'test_t':
        print("输出目标域检测数据集名称，应该是目标域：", args.dataset)
        print("输出目标域检测数据集名称", args.imdbval_name)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    elif args.part == 'test_defocus_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Defocus1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Defocus1, False)
    elif args.part == 'test_defocus_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Defocus1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Defocus3, False)
    elif args.part == 'test_glass_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Glass1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Glass1, False)
    elif args.part == 'test_glass_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Glass3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Glass3, False)
    elif args.part == 'test_motion_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Motion1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Motion1, False)
    elif args.part == 'test_motion_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Motion3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Motion3, False)
    elif args.part == 'test_zoom_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Zoom1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Zoom1, False)
    elif args.part == 'test_zoom_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Zoom3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Zoom3, False)
    elif args.part == 'test_snow_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Snow1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Snow1, False)
    elif args.part == 'test_snow_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Snow3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Snow3, False)
    elif args.part == 'test_frost_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Frost1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Frost1, False)
    elif args.part == 'test_frost_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Frost3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Frost3, False)
    elif args.part == 'test_fog_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Fog1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Fog1, False)
    elif args.part == 'test_fog_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Fog3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Fog3, False)
    elif args.part == 'test_brightness_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Brightness1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Brightness1, False)
    elif args.part == 'test_brightness_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Brightness3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Brightness3, False)
    elif args.part == 'test_contrast_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Contrast1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Contrast1, False)
    elif args.part == 'test_contrast_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Contrast3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Contrast3, False)
    elif args.part == 'test_elastic_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Elastic1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Elastic1, False)
    elif args.part == 'test_elastic_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Elastic3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Elastic3, False)
    elif args.part == 'test_pixelate_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Pixelate1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Pixelate1, False)
    elif args.part == 'test_pixelate_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Pixelate3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Pixelate3, False)
    elif args.part == 'test_jpeg_1':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Jpeg1)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Jpeg1, False)
    elif args.part == 'test_jpeg_3':
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Jpeg3)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Jpeg3, False)
    else:
        print("don't have the test part !")
        pdb.set_trace()



    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    # if not os.path.exists(input_dir):
    #   raise Exception('There is no input directory for loading network from ' + input_dir)
    # load_name = os.path.join(input_dir,
    #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

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
    # fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN = configure_model(fasterRCNN)  # 新增加的,对特征提取层中的所有非bn固定

    for name, value in fasterRCNN.named_parameters():
        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    params,_ = collect_params(fasterRCNN)
    # print ("参数",params)
    if args.optimizer == "adam":
        # lr = lr * 0.1
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
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    fasterRCNN.train()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    # tsne_feats = np.zeros((1,2048))
    # tsne_label = []
    # tsne_info = {}

    time_start = time.time()
    for i in range(num_images):
        # if i>100:
        #     break
        print("第i张测试图",i)
        data = next(data_iter)
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])



        det_tic = time.time()
        rois,_, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label,_,_,_= fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        cls_prob = cls_prob.view( rois.size(1), -1)
        print("维度：", cls_prob.size())
        log_prob = torch.log(cls_prob)
        loss = -torch.sum(cls_prob * log_prob, dim=1).mean()
        # # backward
        optimizer.zero_grad()
        loss.backward()
        # if args.net == "vgg16":
        #     clip_gradient(fasterRCNN, 43.)
        optimizer.step()

        # if i == 1:
        #     macs, params = profile(fasterRCNN, inputs=(im_data, im_info, gt_boxes, num_boxes,))
        #     macs, params = clever_format([macs, params], "%.3f")
        #     print(macs, params)

        #
        # tsne_feats = np.vstack((tsne_feats, pool_feat.detach().cpu().numpy()))
        # tsne_label.append(rois_label.detach().cpu().numpy())

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
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
            # Simply repeat the boxes, once for each class
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
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                #keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # [N, 1]
                cls_dets = cls_dets[keep.view(-1).long()]
                # if vis:
                #     im2show = vis_detections(im2show, imdb.classes[j], j, cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
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

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()
        # if vis:
        #     # 创建保存目录（如果不存在）
        #     save_dir = os.path.join(output_dir, 'vis_results')
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #
        #     # 保存图片，使用图片索引命名
        #     save_path = os.path.join(save_dir, f'{i:06d}_TENT.png')
        #     cv2.imwrite(save_path, im2show)
        #     print(f"已保存可视化结果: {save_path}")
    end = time.time()
    each_time = (end - time_start)/num_images
    print("fps time: %0.4fs" % (each_time))
    print("fps: %0.2f" % (1.0/each_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    # ============== 计算混淆矩阵并生成图片（只包含前景类别） ==============
    print("\n" + "=" * 80)
    print("计算混淆矩阵（只包含前景类别）...")
    print("=" * 80)

    # 获取GT roidb
    gt_roidb = roidb

    # 计算混淆矩阵（只包含前景类别）
    conf_matrix, foreground_class_names = compute_confusion_matrix_no_background(
        all_boxes, imdb, gt_roidb, imdb.num_classes, iou_thresh=0.5
    )

    print(f"前景类别数量: {len(foreground_class_names)}")
    print(f"前景类别: {foreground_class_names}")

    # 保存文本格式的混淆矩阵
    # save_confusion_matrix_foreground(conf_matrix, foreground_class_names, output_dir)

    # 使用PIL绘制混淆矩阵图片（原始数值）
    # plot_confusion_matrix_pil_foreground(conf_matrix, foreground_class_names, output_dir, normalize=False)

    # 使用PIL绘制归一化混淆矩阵图片
    # plot_confusion_matrix_pil_foreground(conf_matrix, foreground_class_names, output_dir, normalize=True)

    # print("\n混淆矩阵生成完成！")