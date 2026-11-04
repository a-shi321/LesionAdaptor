# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.cityscape import cityscape
from datasets.pascal_voc import pascal_voc
from datasets.watercolor import watercolor
from datasets.wildlife import wildlife
from datasets.comic import comic
from datasets.xray8 import xray8
from datasets.vindr import vindr
from datasets.vindr_merge import vindr_merge
from datasets.polygen import polygen
from datasets.brainA import brainA
from datasets.brainB import brainB
from datasets.brainC import brainC
from datasets.IDRiD import IDRiD
from datasets.FGADR import FGADR
from datasets.cvc import cvc
from datasets.etis import etis
from datasets.kvasir import kvasir
###########################################
for year in ['2007']:
  for split in ['train_s', 'test_t','test_s', 'train_t', 'train_cg_fg']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (
      lambda split=split, year=year:
      print("CALL:", split, year) or cityscape(split, year)
    )
    print(name, year, split)
    # print("这是啥？",__sets[name])
for year in ['2007']:
  for split in ['train', 'test_s', 'test','Day','Night','Season_Spring','Season_SummerFall','Season_Winter','Weather_Fog','Weather_RainSnow','Weather_Sunny',
                'test_defocus_blur_1',  'test_glass_blur_1',  'test_motion_blur_1',  'test_zoom_blur_1',  'test_snow_1', 'test_frost_1', 'test_fog_1',    'test_brightness_1',  'test_contrast_1', 'test_elastic_transform_1', 'test_pixelate_1', 'test_jpeg_compression_1',
                'test_defocus_blur_3',  'test_glass_blur_3',  'test_motion_blur_3',  'test_zoom_blur_3',  'test_snow_3', 'test_frost_3', 'test_fog_3',    'test_brightness_3',  'test_contrast_3', 'test_elastic_transform_3', 'test_pixelate_3', 'test_jpeg_compression_3']:
    name = 'wildlife_{}_{}'.format(year, split)
    # __sets[name] = (lambda split=split : wildlife(split,year))
    __sets[name] = (lambda split=split, year=year: print("CALL:", split, year) or wildlife(split, year)
    )
    print(name, year, split)
for year in ['2007', '2012']:
  for split in ['trainval', 'test']:
    name = 'VOC_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['trainval', 'test']:
    name = 'watercolor_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: watercolor(split, year))
for year in ['2007', '2015']:
  for split in ['trainval', 'test']:
    name = 'comic_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: comic(split, year))

for year in ['2007', '2012']:
  for split in ['train_aug', 'test','train', 'test_aug']:
    name = 'xray8_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: xray8(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'BrainA_source_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: brainA(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'BrainB_target_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: brainB(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'BrainC_target_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: brainC(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'vindr_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: vindr(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test','test_gaussian_noise','test_defocus_blur','test_brightness','test_contrast']:
    name = 'vindr_merge_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: vindr_merge(split, year))


for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'polygen_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: polygen(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'cvc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cvc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test']:
    name = 'etis_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: etis(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'test','train_s','test_s','test_s_gaussian_noise','test_s_defocus_blur','test_s_brightness','test_s_contrast']:
    name = 'kvasir_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: kvasir(split, year))



for year in ['2007', '2012']:
  for split in ['train_aug', 'test_aug','train_aug_without_min','test_aug_without_min']:
    name = 'IDRiD_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: IDRiD(split, year))
for year in ['2007', '2012']:
  for split in ['train_aug', 'test_aug','train','test']:
    name = 'FGADR_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: FGADR(split, year))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
