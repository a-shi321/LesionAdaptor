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


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='cityscape', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--load_dir', dest='load_dir',
    #                     help='directory to load models', default="models",
    #                     type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to load models', default="models.pth",
                        type=str)
    parser.add_argument('--part', dest='part',
                        help='test_s or test_t or test_all', default="test_t",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--model_name', dest='model_name',
                        help='model file name',
                        default='res101.bs1.pth', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def collect_params(model):
    chosen_layers = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # print ("开始获取参数")
            chosen_layers.append(m.running_mean)
    return chosen_layers

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "VOC_2007_trainval"
      args.t_imdbtest_name = "VOC_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_t"
      args.s_imdbtest_name = "cityscape_2007_test_s"
      args.t_imdbtest_name = "cityscape_2007_test_t"
      args.all_imdbtest_name = "cityscape_2007_test_all"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "VOC_test":
      args.t_imdb_name = "VOC_2007_train_trainval+VOC_2012_train_trainval"
      # args.s_imdb_name = "watercolor_2007_train"
      args.t_imdbtest_name = "VOC_2007_test"
      # args.s_imdbtest_name = "watercolor_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "wildlife":
     args.imdb_name = "wildlife_2007_train"
     args.s_imdbtest_name = "wildlife_2007_test_s"
     args.t_imdbtest_name = "wildlife_2007_test"
     args.t_imdbtest_name_Day = "wildlife_2007_Day"
     args.t_imdbtest_name_Night = "wildlife_2007_Night"
     args.t_imdbtest_name_Season_Spring = "wildlife_2007_Season_Spring"
     args.t_imdbtest_name_Season_SummerFall = "wildlife_2007_Season_SummerFall"
     args.t_imdbtest_name_Season_Winter = "wildlife_2007_Season_Winter"
     args.t_imdbtest_name_Weather_Fog = "wildlife_2007_Weather_Fog"
     args.t_imdbtest_name_Weather_RainSnow = "wildlife_2007_Weather_RainSnow"
     args.t_imdbtest_name_Weather_Sunny = "wildlife_2007_Weather_Sunny"
     args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "VOC2water":
     print('loading our dataset........，，...')
     args.s_imdb_name = "VOC_2007_train_trainval"  # 用
     args.t_imdb_name = "watercolor_2007_train"  # 用city中的source作为target
     args.s_imdbtest_name = "VOC_2007_test"
     args.t_imdbtest_name = "watercolor_2007_test"
     args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "VOC2comic":
      print('loading our dataset...........')
      args.s_imdb_name = "VOC_2007_train_trainval"  # 用
      args.t_imdb_name = "watercolor_2007_train"  # 用city中的source作为target
      args.s_imdbtest_name = "VOC_2007_test"
      args.t_imdbtest_name = "comic_2015_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "xray8":
      print('loading our dataset...........')
      args.imdb_name = "xray8_2007_train_aug"
      args.t_imdbtest_name = "xray8_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

  #
  # args = set_dataset_args(args,test=True)
  # if torch.cuda.is_available() and not args.cuda:
  #   print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  # np.random.seed(cfg.RNG_SEED)

  # if args.cfg_file is not None:
  #   cfg_from_file(args.cfg_file)
  # if args.set_cfgs_target is not None:
  #   cfg_from_list(args.set_cfgs_target)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  if args.part == 'test_s':
      print("输出目标域检测数据集名称", args.s_imdbtest_name)
      imdb, roidb, ratio_list, ratio_index = combined_roidb(args.s_imdbtest_name, False)
  elif args.part == 'test_t':
      print("输出目标域检测数据集名称，应该是目标域：", args.dataset)
      print("输出目标域检测数据集名称", args.t_imdbtest_name)
      imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name, False)
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

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  # from model.faster_rcnn.vgg16 import vgg16


  # initilize the network here.
  from model.faster_rcnn.vgg16 import vgg16_test
  from model.faster_rcnn.resnet import resnet

  if args.net == 'vgg16':
      fasterRCNN = vgg16_test(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
      fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
      fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  load_name = args.model_dir
  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
  params = collect_params(fasterRCNN)
  print("纯测试获取参数",params[-1])
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
  max_per_image = 100

  thresh = 0.0


  # save_name = args.load_name.split('/')[-1]
  #save_name = args.log_ckpt_name + '_test_in_'+ args.dataset_t
  save_name = args.part + '_faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  #print("test_net的输出",output_dir)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize=False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      im_data.resize_(data[0].size()).copy_(data[0])
      #print(data[0].size())
      im_info.resize_(data[1].size()).copy_(data[1])
      gt_boxes.resize_(data[2].size()).copy_(data[2])
      num_boxes.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois,cls_score, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label= fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      # d_pred = d_pred.data
      # path = data[4]

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

      scores = scores.squeeze() #[1, 300, 2] -> [300, 2]
      pred_boxes = pred_boxes.squeeze() #[1, 300, 8] -> [300, 8]
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()

      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1) #[300]
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds] #[300]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4] #[300, 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1) # [300, 5]
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS) #[N, 1]
            cls_dets = cls_dets[keep.view(-1).long()]#[N, 5]

            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])#[M,]
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



  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
