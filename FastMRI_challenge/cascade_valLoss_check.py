import torch
import argparse
import shutil
import os, sys
from pathlib import Path

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, MS_SSIM_L1_LOSS
from utils.model.varnet import VarNet

import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
import utils.model.fastmri

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

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
    parser.add_argument('-v', '--data-path-val', type=Path, default='/home/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=20, help='Number of cascades | Should be less than 12') ## important hyperparameter
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
    args = parser.parse_args()
    return args

def cascadeOutput(model, masked_kspace, mask, cascade_num):
    sens_maps = model.sens_net(masked_kspace, mask)
    kspace_pred = masked_kspace.clone()

    for i in range(cascade_num):
        kspace_pred = model.cascade[i](kspace_pred, masked_kspace, mask, sens_maps)
    result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
    height = result.shape[-2]
    width = result.shape[-1]
    return result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]



def validate(args, model, data_loader, cascade):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # Apply gradient checkpointing
            output = cascadeOutput(model, kspace, mask, cascade)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def load_checkpoint(exp_dir, model, optimizer):
    checkpoint_path = exp_dir / 'model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, with best validation loss {best_val_loss:.4g}.")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
    return start_epoch, best_val_loss

        
def valLossChecking(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    # loss_type = SSIMLoss().to(device=device)
    loss_type = MS_SSIM_L1_LOSS().to(device=device)  # 새로 만든 MS_SSIM_L1_LOSS 사용
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.RAdam(model.parameters(), args.lr)  # RAdam optimizer 사용

    # Check if a checkpoint exists, and load it if it does
    start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model, optimizer)

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args
    )

    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")

    # 기존 val_loss_log 파일이 존재하면 불러오기, 없으면 빈 배열 생성
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))
    print(val_loss_log)
    for casca in range(1, args.cascade+1):
        print(f'Cascade#01~ #{casca:2d} ............... {args.net_name} ...............')
        
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, casca)
        
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        print("complete")
        print(
            f'Cascade = [{casca:4d}/{args.cascade:4d}]'
            f'ValLoss = {val_loss:.4g} ValTime = {val_time:.4f}s',
        )

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.mask_aug_on = True
    args.gradient_accumulation_steps = 10
    args.max_epochs = args.num_epochs

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    valLossChecking(args)

    