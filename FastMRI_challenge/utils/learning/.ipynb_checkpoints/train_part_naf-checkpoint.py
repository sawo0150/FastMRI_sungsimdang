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

from collections import defaultdict
from utils.data.load_data_naf import create_data_loaders_naf
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, MS_SSIM_L1_LOSS, MS_SSIM_L1_LOSS2

# from utils.model.varnet import VarNet
# from promptMR.pl_modules.promptmr_module import PromptMrModule  # PromptMR 모델을 가져옵니다.
from promptMR.models.promptmr import PromptMR
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet

# DataAugmentor와 관련된 import 추가
from MRAugment.mraugment.data_augment import DataAugmentor
from MaskAugment.maskAugment import MaskAugmentor

import os
import math
from torch.optim.lr_scheduler import _LRScheduler
# MaskAugmentor와 관련된 코드 추가
import random

from torch.cuda.amp import autocast, GradScaler
import gc
import fastmri

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False


scaler = GradScaler()  # Mixed Precision을 위한 GradScaler 초기화

def checkpointed_forward(module, *inputs):
    # 모든 입력 텐서가 requires_grad=True 상태인지 확인합니다.
    inputs = [inp.float().requires_grad_(True) if torch.is_tensor(inp) and inp.is_floating_point() else inp for inp in inputs]
    return checkpoint.checkpoint(module, *inputs)

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, augmentor, mask_augmentor):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        koutput, grappa, iinput, target, maximum, _, _ = data
        # print("train_epoch 함수 : ", kspace.shape)
        # print("mask 함수 : ", mask.shape)
        
        koutput = koutput.cuda(non_blocking=True).requires_grad_(True)  # kspace는 grad 필요
        grappa = grappa.cuda(non_blocking=True).requires_grad_(True)
        iinput = iinput.cuda(non_blocking=True).requires_grad_(True)
        target = target.cuda(non_blocking=True).requires_grad_(False)  # target은 grad 필요 없음
        maximum = maximum.cuda(non_blocking=True).requires_grad_(False)  # maximum도 마찬가지

        input_combined = torch.cat((koutput, grappa, iinput), dim=1) # 3개의 채널을 가진 이미지로 만듦

        with autocast():  # Mixed Precision 사용
            # Apply gradient checkpointings
            # output = checkpointed_forward(model, kspace, mask)
            output = model(input_combined)

            loss = loss_type(output, target, maximum)
            loss = loss / args.gradient_accumulation_steps  # Scale the loss for gradient accumulation

        scaler.scale(loss).backward()


        if (iter + 1) % args.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs_naf:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * args.gradient_accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch

def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            koutput, grappa, iinput, target, _, fnames, slices = data
            koutput = koutput.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            iinput = iinput.cuda(non_blocking=True)

            input_combined = torch.cat((koutput, grappa, iinput), dim=1)

            # Apply gradient checkpointing
            # output = checkpointed_forward(model, kspace, mask)
            output = model(input_combined)

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

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, model_filename='model_naf.pt'):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / model_filename
    )
    if is_new_best:
        shutil.copyfile(exp_dir / model_filename, exp_dir / f'best_{model_filename}')

def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

def load_checkpoint(exp_dir, model, optimizer, model_filename='model_naf.pt'):
    checkpoint_path = exp_dir / model_filename
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

def train3(args):
    clear_gpu_memory()
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # NafNet 모델을 사용하도록 변경
    model = NAFNet(
        img_channel=args.img_channel_naf,
        width=args.width_naf,
        middle_blk_num=args.middle_blk_num_naf,
        enc_blk_nums=args.enc_blk_nums_naf,
        dec_blk_nums=args.dec_blk_nums_naf
    )
    model.to(device=device)

    # loss_type = SSIMLoss().to(device=device)
    loss_type = MS_SSIM_L1_LOSS().to(device=device)  # 새로 만든 MS_SSIM_L1_LOSS 사용
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.RAdam(model.parameters(), args.lr_naf)  # RAdam optimizer 사용

    val_loader = create_data_loaders_naf(
        data_path=args.data_path_val,
        args=args
    )
    
    # Check if a checkpoint exists, and load it if it does
    start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model, optimizer)

    train_loader = create_data_loaders_naf(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmentor=None,      # augmentor가 None이면 전달되지 않음
        mask_augmentor=None  # mask_augmentor가 None이면 전달되지 않음
    )

    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log3.npy")

    # 기존 val_loss_log 파일이 존재하면 불러오기, 없으면 빈 배열 생성
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))
    print(val_loss_log)
    for epoch in range(start_epoch, args.num_epochs_naf):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, None, None)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        # train_loss = 0 
        # train_time = 0 
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        print("complete")
        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs_naf:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )


        # # 모델 평가를 위한 외부 스크립트 실행 전에 GPU 메모리 정리
        # torch.cuda.empty_cache()

        # # **모델 평가를 위한 외부 스크립트 실행**
        # reconstruct_cmd = f"python3 reconstruct.py -b 2 -n '{args.net_name}' -p '/home/swpants05/fastmri-2024-data/leaderboard'"
        # os.system(reconstruct_cmd)

        # eval_cmd = f"python3 leaderboard_eval.py -lp '/home/swpants05/fastmri-2024-data/leaderboard' -yp '/home/swpants05/fastMRISungsimdang_ws/root_sungsimV1/result/{args.net_name}/reconstructions_leaderboard'"
        # os.system(eval_cmd)

        # print(f"Reconstruction and evaluation done for epoch {epoch}.")



def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def filter_optimizer_state_dict(optimizer_state_dict, model):
    """필요하지 않은 파라미터 그룹을 옵티마이저 상태에서 제거."""
    new_state_dict = optimizer_state_dict.copy()
    param_ids = {id(p) for p in model.parameters() if p.requires_grad}
    
    new_groups = []
    for group in optimizer_state_dict['param_groups']:
        new_params = [p_id for p_id in group['params'] if p_id in param_ids]
        if new_params:
            group['params'] = new_params
            new_groups.append(group)

    new_state_dict['param_groups'] = new_groups
    return new_state_dict
