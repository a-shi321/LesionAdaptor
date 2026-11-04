import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='pascal_voc_0712', type=str)
    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='clipart', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5, type=float)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for pixel level',
                        action='store_true')
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',default=True,
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--part', dest='part',
                        help='test_s or test_t or test_all', default="test_t",
                        type=str)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    args = parser.parse_args()
    return args

def set_dataset_args(args, test=False):
    if not test:
        print("这部分？")
        if args.dataset == "cityscape":
            print('loading our cityscape dataset...........')
            args.imdb_name = "cityscape_2007_train_s"
            args.imdbval_s_name = "cityscape_2007_train_s"
            args.imdbval_name = "cityscape_2007_test_t"
            args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "wildlife":
            print('loading our wildlife dataset...........')
            args.imdb_name = "wildlife_2007_train"
            args.imdbval_name = "wildlife_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

        elif args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_water":
            args.imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
            args.imdbval_name = "voc_clipart_2007_trainval+voc_clipart_2012_trainval"
            args.imdb_name_cycle = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cycleclipart":
            args.imdb_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.imdbval_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cyclewater":
            args.imdb_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.imdbval_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.imdb_name_cycle = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_trainval"
            args.imdbval_name = "foggy_cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "vg":
            args.imdb_name = "vg_150-50-50_minitrain"
            args.imdbval_name = "vg_150-50-50_minival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_trainval"
            args.imdbval_name = "cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_train"
            args.imdbval_name = "sim10k_train"
            args.imdb_name_cycle = "sim10k_cycle_train"  # "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "sim10k_cycle":
            args.imdb_name = "sim10k_cycle_train"
            args.imdbval_name = "sim10k_cycle_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        ## cityscape dataset for only car classes.
        # elif args.dataset == "cityscape_kitti":
        #     args.imdb_name = "cityscape_kitti_trainval"
        #     args.imdbval_name = "cityscape_kitti_trainval"
        #     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                      '30']
        elif args.dataset_t == "water":
            args.imdb_name_target = "water_train"
            args.imdbval_name_target = "water_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"
            args.imdbval_name_target = "clipart_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_trainval"
            args.imdbval_name_target = "cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        ## cityscape dataset for only car classes.
        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_trainval"
            args.imdbval_name_target = "cityscape_car_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        # elif args.dataset_t == "kitti":
        #     args.imdb_name_target = "kitti_trainval"
        #     args.imdbval_name_target = "kitti_trainval"
        #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                             '20']
        elif args.dataset_t == "foggy_cityscape":
            args.imdb_name_target = "foggy_cityscape_trainval"
            args.imdbval_name_target = "foggy_cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
        elif args.dataset == "brain":
            args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainA_source_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "brainB":
            #args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainB_target_2007_test"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "brainC":
            #args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainC_target_2007_test"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    else:
        print("还是这部分？")
        if args.dataset == "cityscape":
            print('loading our cityscape dataset...........')
            args.imdb_name = "cityscape_2007_train_s"
            args.imdbval_name = "cityscape_2007_test_t"
            args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "wildlife":
            print('loading our wildlife dataset...........')
            args.imdb_name = "wildlife_2007_train"
            args.imdbval_name = "wildlife_2007_test"
            args.s_imdbtest_name="wildlife_2007_test_s"
            args.t_imdbtest_name_Day = "wildlife_2007_Day"
            args.t_imdbtest_name_Night = "wildlife_2007_Night"
            args.t_imdbtest_name_Season_Spring = "wildlife_2007_Season_Spring"
            args.t_imdbtest_name_Season_SummerFall = "wildlife_2007_Season_SummerFall"
            args.t_imdbtest_name_Season_Winter = "wildlife_2007_Season_Winter"
            args.t_imdbtest_name_Weather_Fog = "wildlife_2007_Weather_Fog"
            args.t_imdbtest_name_Weather_RainSnow = "wildlife_2007_Weather_RainSnow"
            args.t_imdbtest_name_Weather_Sunny = "wildlife_2007_Weather_Sunny"
            '''
            'defocus_blur',  'glass_blur',  'motion_blur',  'zoom_blur', 
        'snow', 'frost', 'fog',    'brightness', 
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
            '''
            args.t_imdbtest_name_Defocus1 = "wildlife_2007_test_defocus_blur_1"
            args.t_imdbtest_name_Defocus3 = "wildlife_2007_test_defocus_blur_3"
            args.t_imdbtest_name_Glass1 = "wildlife_2007_test_glass_blur_1"
            args.t_imdbtest_name_Glass3 = "wildlife_2007_test_glass_blur_3"
            args.t_imdbtest_name_Motion1 = "wildlife_2007_test_motion_blur_1"
            args.t_imdbtest_name_Motion3 = "wildlife_2007_test_motion_blur_3"
            args.t_imdbtest_name_Zoom1 = "wildlife_2007_test_zoom_blur_1"
            args.t_imdbtest_name_Zoom3 = "wildlife_2007_test_zoom_blur_3"
            args.t_imdbtest_name_Snow1 = "wildlife_2007_test_snow_1"
            args.t_imdbtest_name_Snow3 = "wildlife_2007_test_snow_3"
            args.t_imdbtest_name_Frost1 = "wildlife_2007_test_frost_1"
            args.t_imdbtest_name_Frost3 = "wildlife_2007_test_frost_3"
            args.t_imdbtest_name_Fog1 = "wildlife_2007_test_fog_1"
            args.t_imdbtest_name_Fog3 = "wildlife_2007_test_fog_3"
            args.t_imdbtest_name_Brightness1 = "wildlife_2007_test_brightness_1"
            args.t_imdbtest_name_Brightness3 = "wildlife_2007_test_brightness_3"
            args.t_imdbtest_name_Contrast1 = "wildlife_2007_test_contrast_1"
            args.t_imdbtest_name_Contrast3 = "wildlife_2007_test_contrast_3"
            args.t_imdbtest_name_Elastic1 = "wildlife_2007_test_elastic_transform_1"
            args.t_imdbtest_name_Elastic3 = "wildlife_2007_test_elastic_transform_3"
            args.t_imdbtest_name_Pixelate1 = "wildlife_2007_test_pixelate_1"
            args.t_imdbtest_name_Pixelate3 = "wildlife_2007_test_pixelate_3"
            args.t_imdbtest_name_Jpeg1 = "wildlife_2007_test_jpeg_compression_1"
            args.t_imdbtest_name_Jpeg3 = "wildlife_2007_test_jpeg_compression_3"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

        elif args.dataset == "xray8":
            print('loading our dataset...........')
            args.imdb_name = "xray8_2007_train_aug"
            args.imdbval_name = "xray8_2007_test_aug"
            args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        # elif args.dataset == "brain":
        #     args.imdb_name = "BrainA_source_2007_train"
        #     args.imdbval_name = "BrainA_source_2007_test"
        #     args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "brain":
            args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainA_source_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "brainB":
            #args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainB_target_2007_test"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "brainC":
            #args.imdb_name = "BrainA_source_2007_train"
            args.imdbval_name = "BrainC_target_2007_test"
            args.imdbval_s_name = "BrainA_source_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']



        elif args.dataset == "vindr":
            args.imdb_name = "vindr_2007_train"
            args.imdbval_name = "vindr_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "vindr_merge":
            args.imdb_name = "vindr_merge_2007_train"
            args.imdbval_name = "vindr_merge_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "vindr_merge_gaussian":#'test_gaussian_noise',
            args.imdb_name = "vindr_merge_2007_train"
            args.imdbval_name = "vindr_merge_2007_test_gaussian_noise"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "vindr_merge_defocus":  # 'test_defocus_blur',
            args.imdb_name = "vindr_merge_2007_train"
            args.imdbval_name = "vindr_merge_2007_test_defocus_blur"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]',
                             'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "vindr_merge_brightness":  # 'test_brightness',
            args.imdb_name = "vindr_merge_2007_train"
            args.imdbval_name = "vindr_merge_2007_test_brightness"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]',
                             'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "vindr_merge_contrast":  # 'test_contrast'
            args.imdb_name = "vindr_merge_2007_train"
            args.imdbval_name = "vindr_merge_2007_test_contrast"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32,64,128,256]', 'ANCHOR_RATIOS', '[0.5,1,2]',
                             'MAX_NUM_GT_BOXES', '50']


        elif args.dataset == "polygen":
            args.imdb_name = "polygen_2007_train"
            args.imdbval_name = "polygen_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "cvc":
            args.imdb_name = "cvc_2007_train"
            args.imdbval_name = "cvc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "etis":
            args.imdb_name = "etis_2007_train"
            args.imdbval_name = "etis_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "kvasir":
            args.imdb_name = "kvasir_2007_train"
            args.imdbval_name = "kvasir_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "kvasir_gaussian":
            args.imdb_name = "kvasir_2007_train"
            args.imdbval_name = "kvasir_2007_test_s_gaussian_noise"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "kvasir_defocus":
            args.imdb_name = "kvasir_2007_train"
            args.imdbval_name = "kvasir_2007_test_s_defocus_blur"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "kvasir_brightness":
            args.imdb_name = "kvasir_2007_train"
            args.imdbval_name = "kvasir_2007_test_s_brightness"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "kvasir_contrast":
            args.imdb_name = "kvasir_2007_train"
            args.imdbval_name = "kvasir_2007_test_s_contrast"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']






        elif args.dataset == "IDRiD":
            args.imdb_name = "IDRiD_2007_train_aug_without_min"
            args.imdbval_name = "IDRiD_2007_test_aug_without_min"
            args.set_cfgs = ['ANCHOR_SCALES', '[1,2,4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "FGADR":
            args.imdb_name = "FGADR_2007_train_aug"
            args.imdbval_name = "FGADR_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[1,2,4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','50']




        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_test"
            args.imdbval_name = "foggy_cityscape_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "cityscape_kitti":
            args.imdb_name = "cityscape_kitti_val"
            args.imdbval_name = "cityscape_kitti_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "water":
            args.imdb_name = "water_test"
            args.imdbval_name = "water_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "clipart":
            args.imdb_name = "clipart_trainval"
            args.imdbval_name = "clipart_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_val"
            args.imdbval_name = "cityscape_car_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args
