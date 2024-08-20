import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part_naf import train_naf

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    # DataAugmentor 관련 argument 추가
    parser.add_argument('--aug-on', default=False, action='store_true', help='Switch to turn data augmentation on')
    parser.add_argument('--aug-schedule', type=str, default='exp', choices=['constant', 'ramp', 'exp'], help='Type of augmentation scheduling')
    parser.add_argument('--aug-delay', type=int, default=0, help='Number of epochs without data augmentation at the start of training')
    parser.add_argument('--aug-strength', type=float, default=0.3, help='Maximum augmentation strength')
    parser.add_argument('--aug-exp-decay', type=float, default=5.0, help='Exponential decay coefficient for augmentation scheduling')
    parser.add_argument('--aug-interpolation-order', type=int, default=1, help='Interpolation order used in augmentation, 1: bilinear, 3: bicubic')
    parser.add_argument('--aug-upsample', default=False, action='store_true', help='Upsample before augmentation to avoid aliasing artifacts')
    parser.add_argument('--aug-upsample-factor', type=int, default=2, help='Upsample factor before augmentation')
    parser.add_argument('--aug-upsample-order', type=int, default=1, help='Order of upsampling filter before augmentation, 1: bilinear, 3: bicubic')

    # MRPromptModel 관련 argument 추가
    parser.add_argument('--cascade', type=int, default=6, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--num_adj_slices', type=int, default=1, help='Number of adjacent slices')
    parser.add_argument('--n_feat0', type=int, default=32, help='Number of top-level channels for PromptUnet')
    parser.add_argument('--feature_dim', nargs='+', type=int, default=[48, 64, 80], help='Feature dimensions')
    parser.add_argument('--prompt_dim', nargs='+', type=int, default=[16, 32, 48], help='Prompt dimensions')
    parser.add_argument('--sens_n_feat0', type=int, default=16, help='Initial number of channels for sensitivity Unet')
    parser.add_argument('--sens_feature_dim', nargs='+', type=int, default=[24, 32, 40], help='Feature dimensions for sensitivity Unet')
    parser.add_argument('--sens_prompt_dim', nargs='+', type=int, default=[8, 16, 24], help='Prompt dimensions for sensitivity Unet')
    parser.add_argument('--len_prompt', nargs='+', type=int, default=[5, 5, 5], help='Length of prompt')
    parser.add_argument('--prompt_size', nargs='+', type=int, default=[32, 16, 8], help='Size of prompt')
    parser.add_argument('--n_enc_cab', nargs='+', type=int, default=[2, 3, 3], help='Number of encoder CABs')
    parser.add_argument('--n_dec_cab', nargs='+', type=int, default=[2, 2, 3], help='Number of decoder CABs')
    parser.add_argument('--n_skip_cab', nargs='+', type=int, default=[1, 1, 1], help='Number of skip CABs')
    parser.add_argument('--n_bottleneck_cab', type=int, default=3, help='Number of bottleneck CABs')
    parser.add_argument('--no_use_ca', action='store_true', help='Disable channel attention')
    parser.add_argument('--use_checkpoint',type=bool, default=True, help='Use gradient checkpointing to save memory')
    parser.add_argument('--low_mem',type=bool, default=True, help='Use low memory settings')


    parser.add_argument('--aug-weight-translation', type=float, default=0, help='Weight of translation probability')
    parser.add_argument('--aug-weight-rotation', type=float, default=1.0, help='Weight of arbitrary rotation probability')
    parser.add_argument('--aug-weight-shearing', type=float, default=0, help='Weight of shearing probability')
    parser.add_argument('--aug-weight-scaling', type=float, default=1.0, help='Weight of scaling probability')
    parser.add_argument('--aug-weight-rot90', type=float, default=1.0, help='Weight of rotation by multiples of 90 degrees probability')
    parser.add_argument('--aug-weight-fliph', type=float, default=1.0, help='Weight of horizontal flip probability')
    parser.add_argument('--aug-weight-flipv', type=float, default=1.0, help='Weight of vertical flip probability')

    parser.add_argument('--aug-max-translation-x', type=float, default=0.125, help='Maximum translation along the x axis as a fraction of image width')
    parser.add_argument('--aug-max-translation-y', type=float, default=0.125, help='Maximum translation along the y axis as a fraction of image height')
    parser.add_argument('--aug-max-rotation', type=float, default=180., help='Maximum rotation applied in either direction (degrees)')
    parser.add_argument('--aug-max-shearing-x', type=float, default=15.0, help='Maximum shearing along the x axis (degrees)')
    parser.add_argument('--aug-max-shearing-y', type=float, default=15.0, help='Maximum shearing along the y axis (degrees)')
    parser.add_argument('--aug-max-scaling', type=float, default=0.15, help='Maximum scaling as a fraction of image dimensions')

    # max_train_resolution 인자를 추가합니다.
    parser.add_argument("--max_train_resolution",nargs="+",default=None,type=int,help="If given, training slices will be center cropped to this size if larger along any dimension.")
    
    parser.add_argument('--img_channel_naf', type=int, default=3, help='Number of image channels for NAFNet')
    parser.add_argument('--width_naf', type=int, default=32, help='Width for NAFNet')
    parser.add_argument('--middle_blk_num_naf', type=int, default=1, help='Number of middle blocks for NAFNet')
    parser.add_argument('--enc_blk_nums_naf', type=list, default=[1,1,1,28], help='Encoder block numbers for NAFNet')
    parser.add_argument('--dec_blk_nums_naf', type=list, default=[1,1,1,1], help='Decoder block numbers for NAFNet')
    parser.add_argument('--num_epochs_naf', type=int, default=20, help='Number of epochs for NAFNet training')
    parser.add_argument('--lr_naf', type=float, default=1e-4, help='Learning rate for NAFNet')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.mask_aug_on = True
    args.gradient_accumulation_steps = 10
    args.max_epochs = args.num_epochs

    args.update_interval_rate = 2
    args.num_workers = 8

    args.second_epochs = 70
    args.pre_cascade = 7
    args.second_cascade = 10
    args.second_epoch_list = [126, 250, 400, 500, 600]

    args.additional_cascade_block = 1

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    
    print('train_naf 시작')
    train_naf(args)
    print('train_naf 끝')