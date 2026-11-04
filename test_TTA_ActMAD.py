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
import copy

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.cat(self.outputs,dim=0)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.cat(self.outputs,dim=0)
        out = torch.var(out, dim=0)
        return out

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    print("hello BN")
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    # model.requires_grad_(False)
    for m in model.modules():
        for p in m.parameters(): p.requires_grad = False
    # configure norm for tent updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and ('RCNN_base1' in nm or 'RCNN_base2' in nm):
            print("看看BN层")
            m.eval()
            m.requires_grad=True
            for p in m.parameters(): p.requires_grad = True
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

    source_args = copy.deepcopy(args)
    # source_args.dataset ="brain"


    print("输出看看：",source_args)

    args = set_dataset_args(args, test=True)
    source_args = set_dataset_args(source_args, test=False)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    if args.dataset == "cityscape":
        source_args.imdbval_s_name ="cityscape_2007_train_s"
    elif args.dataset=="brainB" or args.dataset=="brainC":
         source_args.imdbval_s_name="BrainA_source_2007_train"
    elif args.dataset=="vindr_merge_gaussian" or args.dataset=="vindr_merge_defocus" or args.dataset=="vindr_merge_brightness" or args.dataset=="vindr_merge_contrast":
         source_args.imdbval_s_name="vindr_merge_2007_train"
    elif args.dataset=="cvc" or args.dataset=="etis" or args.dataset=="kvasir" :
         source_args.imdbval_s_name="polygen_2007_train"
    elif args.dataset=="kvasir_gaussian" or args.dataset=="kvasir_defocus" or args.dataset=="kvasir_brightness" or args.dataset=="kvasir_contrast" :
         source_args.imdbval_s_name="kvasir_2007_train_s"

    cfg.TRAIN.USE_FLIPPED = False
    imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(source_args.imdbval_s_name, False)
    imdb_s.competition_mode(on=True)

    print('需要统计的源域训练数据的个数{:d} roidb entries'.format(len(roidb_s)))

    # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    # if not os.path.exists(input_dir):
    #   raise Exception('There is no input directory for loading network from ' + input_dir)
    # load_name = os.path.join(input_dir,
    #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    from model.TTA_faster_rcnn.resnet import resnet
    from model.TTA_faster_rcnn_vgg.vgg16 import vgg16

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_s.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_s.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)

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
    # initilize the tensor holder here.
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
    num_images_s = len(imdb_s.image_index)

    dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, 1,imdb_s.num_classes, training=False, normalize=False)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)

    data_iter_s = iter(dataloader_s)

    chosen_bn_layers = []
    for m in fasterRCNN.modules():
        if isinstance(m, nn.BatchNorm2d):
            chosen_bn_layers.append(m)

    chosen_bn_layers = chosen_bn_layers[-10:]
    n_chosen_layers = len(chosen_bn_layers)
    save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]
    clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
    clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
    clean_mean_list_final = []
    clean_var_list_final = []

    for i in range(num_images_s):
        print("正在统计源域的第i个图片",i)
    # for i in range(1000):
        try:
            data = next(data_iter_s)
        except:
            continue
        with torch.no_grad():
            fasterRCNN.eval()
            hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]

            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).zero_()
            num_boxes.resize_(data[3].size()).zero_()

            (rois,_,cls_prob,bbox_pred,rpn_loss_cls_s_fake,rpn_loss_box_s_fake,RCNN_loss_cls_s_fake,RCNN_loss_bbox_s_fake,rois_label_s_fake,_,_,_) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            for j in range(n_chosen_layers):
                clean_mean_act_list[j].update(save_outputs[j].get_out_mean())  # compute mean from clean data
                clean_var_act_list[j].update(save_outputs[j].get_out_var())  # compute variane from clean data

                save_outputs[j].clear()
                hook_list[j].remove()

    for i in range(n_chosen_layers):
        clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
        clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]


    print("源域数据信息统计完毕")
    fasterRCNN.train()
    fasterRCNN = configure_model(fasterRCNN)

    # lr = (1e-4) /30
    lr = 1e-4
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():

        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    save_name = args.load_name.split('/')[-1]

    # for Corrupt_type in range(1,16):

    if True:

        fasterRCNN.load_state_dict(checkpoint['model'], strict=False)

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
        imdb.competition_mode(on=True)
        num_images = len(imdb.image_index)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,imdb.num_classes, training=False, normalize=False)
        TTAdataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0,
                                                 pin_memory=True)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        data_iter = iter(TTAdataloader)
        _t = {'im_detect': time.time(), 'misc': time.time()}
        output_dir = get_output_dir(imdb, save_name)
        det_file = os.path.join(output_dir, 'detections.pkl')
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        time_start = time.time()
        for img_index in range(num_images):
            print("正在测试第i张图片：",img_index)
            try:
                data = next(data_iter)
            except:
                continue

            fasterRCNN.train()
            for m in fasterRCNN.modules():
                for nm, m in fasterRCNN.named_modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            # fasterRCNN = configure_model(fasterRCNN)

            optimizer.zero_grad()
            save_outputs_tta = [SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                             for x in range(n_chosen_layers)]

            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).zero_()
            num_boxes.resize_(data[3].size()).zero_()

            (rois,_,cls_prob,bbox_pred,rpn_loss_cls_s_fake,rpn_loss_box_s_fake,RCNN_loss_cls_s_fake,RCNN_loss_bbox_s_fake,rois_label_s_fake,_,_,_) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
            batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()

            for i in range(n_chosen_layers):
                loss_mean += torch.nn.functional.l1_loss(batch_mean_tta[i].cuda(), clean_mean_list_final[i].cuda())
                loss_var += torch.nn.functional.l1_loss(batch_var_tta[i].cuda(), clean_var_list_final[i].cuda())

            loss = loss_mean + loss_var
            # if img_index%200 == 1:
            print(loss.item())

            loss.backward()
            optimizer.step()
            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hook_list_tta[z].remove()


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
            misc_tic = time.time()

            if vis:
                im = cv2.imread(imdb.image_path_at(img_index))
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
                    # keep = nms(cls_dets, cfg.TEST.NMS)
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # [N, 1]
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], j, cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][img_index] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][img_index] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][img_index][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][img_index][:, -1] >= image_thresh)[0]
                        all_boxes[j][img_index] = all_boxes[j][img_index][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                             .format(img_index + 1, num_images, nms_time))
            sys.stdout.flush()
            if vis:
                # 创建保存目录（如果不存在）
                save_dir = os.path.join(output_dir, 'vis_results')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存图片，使用图片索引命名
                save_path = os.path.join(save_dir, f'{img_index:06d}_ActMAD.png')
                cv2.imwrite(save_path, im2show)
                print(f"已保存可视化结果: {save_path}")
        print('ActMAD TTA Done!')

        end = time.time()
        print("test time: %0.4fs" % (end - time_start))
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)