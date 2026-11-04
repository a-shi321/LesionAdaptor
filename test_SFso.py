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

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

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
    # print("测试数据集是哪个？",args.imdbval_name)
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








    elif args.part == 'test_t_day':
        print("输出目标域检测数据集名称，应该是Day：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Day)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Day, False)
    elif args.part == 'test_t_night':
        print("输出目标域检测数据集名称，应该是night：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Night)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Night, False)
    elif args.part == 'test_t_fog':
        print("输出目标域检测数据集名称，应该是fog：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Weather_Fog)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Weather_Fog, False)
    elif args.part == 'test_t_rain':
        print("输出目标域检测数据集名称，应该是RainSnow：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Weather_RainSnow)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Weather_RainSnow, False)
    elif args.part == 'test_t_sunny':
        print("输出目标域检测数据集名称，应该是Sunny：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Weather_Sunny)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Weather_Sunny, False)
    elif args.part == 'test_t_spring':
        print("输出目标域检测数据集名称，应该是spring：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Season_Spring)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Season_Spring, False)
    elif args.part == 'test_t_summer':
        print("输出目标域检测数据集名称，应该是SummerFall：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Season_SummerFall)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Season_SummerFall, False)
    elif args.part == 'test_t_winter':
        print("输出目标域检测数据集名称，应该是Winter：", args.dataset)
        print("输出目标域检测数据集名称", args.t_imdbtest_name_Season_Winter)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name_Season_Winter, False)
    elif args.part == 'test_all':
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.all_imdbtest_name, False)
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
    from model.SF_faster_rcnn.resnet import resnet
    from model.SF_faster_rcnn_vgg.vgg16 import vgg16

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
    fasterRCNN.eval()
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
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label= fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


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

        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                  .format(i + 1, num_images, detect_time, nms_time))
        # sys.stdout.flush()

    end = time.time()
    each_time = (end - time_start)/num_images
    print("fps time: %0.4fs" % (each_time))
    print("fps: %0.2f" % (1.0/each_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
