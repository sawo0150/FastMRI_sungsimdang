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
from utils.data.load_data import create_data_loaders, create_data_loaders2
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, MS_SSIM_L1_LOSS, MS_SSIM_L1_LOSS2

# from utils.model.varnet import VarNet
# from promptMR.pl_modules.promptmr_module import PromptMrModule  # PromptMR 모델을 가져옵니다.
from promptMR.models.promptmr import PromptMR, PromptMR2

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
        mask, kspace, target, maximum, _, _ = data
        # print("train_epoch 함수 : ", kspace.shape)
        # print("mask 함수 : ", mask.shape)
        
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True).requires_grad_(True)  # kspace는 grad 필요
        target = target.cuda(non_blocking=True).requires_grad_(False)  # target은 grad 필요 없음
        maximum = maximum.cuda(non_blocking=True).requires_grad_(False)  # maximum도 마찬가지


        # # DataAugmentor를 사용해 k-space 데이터를 증강
        # # MaskAugmentor를 사용해 mask를 증강

        # if augmentor.aug_on:
        #     kspace, target = augmentor(kspace, target_size=target.shape[-2:])
        #     mask = mask_augmentor.augment(mask, epoch)

        # # Apply gradient checkpointing
        # output = checkpointed_forward(model, kspace, mask)

        # loss = loss_type(output, target, maximum)
        # loss = loss / args.gradient_accumulation_steps  # Scale the loss for gradient accumulation
        # loss.backward()

        with autocast():  # Mixed Precision 사용
            # Apply gradient checkpointing
            # output = checkpointed_forward(model, kspace, mask)
            output = model(kspace, mask)

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
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * args.gradient_accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def train_epoch2(args, epoch, model, data_loader, optimizers, loss_type, augmentor, mask_augmentor):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    

    # Mixed Precision을 위한 GradScaler 초기화
    scalers = [GradScaler() for _ in range(args.second_cascade - args.pre_cascade)]

    # Gradient accumulation을 위한 변수
    iter_counters = [0] * (args.second_cascade - args.pre_cascade)  # 각 cascade에 대한 iteration counter
    # update_intervals 리스트를 생성
    update_intervals = [(args.update_interval_rate ** ((args.second_cascade - args.pre_cascade) - i+2)) for i in range(args.second_cascade - args.pre_cascade)]
    print(update_intervals)
    accum_loss = [0] * (args.second_cascade - args.pre_cascade)  # 각 cascade에 대한 누적 손실

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        # print("train_epoch 함수 : ", kspace.shape)
        # print("mask 함수 : ", mask.shape)
        
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True).requires_grad_(True)  # kspace는 grad 필요
        target = target.cuda(non_blocking=True).requires_grad_(False)  # target은 grad 필요 없음
        maximum = maximum.cuda(non_blocking=True).requires_grad_(False)  # maximum도 마찬가지


        # Freezing된 첫 6개의 cascade를 통해 초기 k-space output 생성
        with torch.no_grad():
            kspace_pred = kspace.clone()
            # Sensitivity map은 한 번만 계산
            sens_map = model.sens_net(kspace, mask)

            for i in range(args.pre_cascade):  # 첫 6개의 cascade
                kspace_pred = model.cascades[i](kspace_pred, kspace, mask, sens_map)

        # 나머지 3개의 cascade를 하나씩 처리하여 학습
        for i, (optimizer, scaler) in enumerate(zip(optimizers, scalers)):
            # Optimizer 초기화

            with autocast():
                # 해당 cascade를 통해 k-space output 생성
                kspace_pred = model.cascades[args.pre_cascade + i](kspace_pred, kspace, mask, sens_map)
                kspace_pred = torch.chunk(kspace_pred, model.num_adj_slices, dim=1)[model.center_slice]

                result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

                # 이미지 크기 조정
                height = result.shape[-2]
                width = result.shape[-1]
                result = result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
                # print(f"Result tensor shape: {result.shape}")
                # print(f"Target tensor shape: {target.shape}")

                # 손실 계산 및 누적
                loss = loss_type(result, target, maximum)
                loss = loss / update_intervals[i]  # Scale the loss for gradient accumulation
                accum_loss[i] += loss.item()
                
            # Mixed Precision에서 Gradient Accumulation
            scaler.scale(loss).backward()
            # scaler.scale(loss).backward(retain_graph=True)


            if (iter + 1) % update_intervals[i] == 0:
                # Optimizer를 사용하여 파라미터 업데이트
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 누적 손실 초기화
                accum_loss[i] = 0

            # 메모리 관리: 역전파에 대한 메모리 사용량 줄이기 위해 중간 gradient 삭제
            torch.cuda.empty_cache()

        
        total_loss += loss.item() * update_intervals[-1]

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * args.gradient_accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch

def train_epoch3(args, epoch, model, data_loader, optimizer, loss_type, augmentor, mask_augmentor):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        # print("train_epoch 함수 : ", kspace.shape)
        # print("mask 함수 : ", mask.shape)
        
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True).requires_grad_(True)  # kspace는 grad 필요
        target = target.cuda(non_blocking=True).requires_grad_(False)  # target은 grad 필요 없음
        maximum = maximum.cuda(non_blocking=True).requires_grad_(False)  # maximum도 마찬가지


        # Freezing된 첫 6개의 cascade를 통해 초기 k-space output 생성
        with torch.no_grad():
            kspace_pred = kspace.clone()
            # Sensitivity map은 한 번만 계산
            sens_map = model.sens_nets[0](kspace, mask)

            for i in range(args.pre_cascade):  # 첫 7개의 cascade
                kspace_pred = model.cascades[i](kspace_pred, kspace, mask, sens_map)

            for i in range(args.additional_cascade_block-1):
                print("addblock>1")
                sens_map = model.sens_nets[i+1](kspace, mask)
                for j in range(6):
                    kspace_pred = model.cascades[args.pre_cascade+i*6+j](kspace_pred, kspace, mask, sens_map)
            
            sens_map = model.sens_nets[args.additional_cascade_block](kspace, mask)


        # 학습할 부분에 대해서만 그라디언트 계산 및 업데이트
        with torch.set_grad_enabled(True):
            with autocast():
                # 해당 cascade를 통해 k-space output 생성
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-6], kspace_pred, kspace, mask, sens_map)
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-5], kspace_pred, kspace, mask, sens_map)
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-4], kspace_pred, kspace, mask, sens_map)
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-3], kspace_pred, kspace, mask, sens_map)
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-2], kspace_pred, kspace, mask, sens_map)
                kspace_pred = checkpointed_forward(model.cascades[args.pre_cascade+args.additional_cascade_block*6-1], kspace_pred, kspace, mask, sens_map)

                # sens_map = model.sens_nets[args.additional_cascade_block](kspace, mask)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-6](kspace_pred, kspace, mask, sens_map)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-5](kspace_pred, kspace, mask, sens_map)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-4](kspace_pred, kspace, mask, sens_map)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-3](kspace_pred, kspace, mask, sens_map)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-2](kspace_pred, kspace, mask, sens_map)
                # kspace_pred = model.cascades[args.pre_cascade+args.additional_cascade_block*6-1](kspace_pred, kspace, mask, sens_map)
                kspace_pred = torch.chunk(kspace_pred, model.num_adj_slices, dim=1)[model.center_slice]

                result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

                # 이미지 크기 조정
                height = result.shape[-2]
                width = result.shape[-1]
                result = result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
                # print(f"Result tensor shape: {result.shape}")
                # print(f"Target tensor shape: {target.shape}")

                # 손실 계산 및 누적
                loss = loss_type(result, target, maximum)
                loss = loss / args.gradient_accumulation_steps # Scale the loss for gradient accumulation
                
            # Mixed Precision에서 Gradient Accumulation
            scaler.scale(loss).backward()
            # scaler.scale(loss).backward(retain_graph=True)


        if (iter + 1) % args.gradient_accumulation_steps == 0:
            # Optimizer를 사용하여 파라미터 업데이트
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 메모리 관리: 역전파에 대한 메모리 사용량 줄이기 위해 중간 gradient 삭제
        torch.cuda.empty_cache()

        
        total_loss += loss.item() * args.gradient_accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
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
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # Apply gradient checkpointing
            # output = checkpointed_forward(model, kspace, mask)
            output = model(kspace, mask)

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


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, model_filename='model06.pt'):
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

def save_model2(args, exp_dir, epoch, model, optimizers, best_val_loss, is_new_best, model_filename='model.pt'):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
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

def load_checkpoint(exp_dir, model, optimizer, model_filename='model06.pt'):
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

def load_checkpoint2(exp_dir, model, optimizers, model_filename='model.pt'):
    checkpoint_path = exp_dir / model_filename
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        
        if optimizers is not None:
            for opt, opt_state in zip(optimizers, checkpoint['optimizers']):
                try:
                    opt.load_state_dict(opt_state)
                except ValueError as e:
                    print(f"Warning: {e}")
                    print("Optimizer state dict contains a parameter group that doesn't match the size of optimizer's group. Ignoring the optimizer state.")
                    # Optimizer의 파라미터 그룹이 불일치할 경우, 새로운 상태로 덮어씁니다.
                    continue

        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, with best validation loss {best_val_loss:.4g}.")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
    return start_epoch, best_val_loss

def train1(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # PromptMR 모델을 사용하도록 변경
    model = PromptMR(
        num_cascades=args.cascade,
        num_adj_slices=args.num_adj_slices,
        n_feat0=args.n_feat0,
        feature_dim = args.feature_dim,
        prompt_dim = args.prompt_dim,
        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim = args.sens_feature_dim,
        sens_prompt_dim = args.sens_prompt_dim,
        len_prompt = args.len_prompt,
        prompt_size = args.prompt_size,
        n_enc_cab = args.n_enc_cab,
        n_dec_cab = args.n_dec_cab,
        n_skip_cab = args.n_skip_cab,
        n_bottleneck_cab = args.n_bottleneck_cab,
        no_use_ca = args.no_use_ca,
        use_checkpoint=args.use_checkpoint,
        low_mem = args.low_mem
    )
    model.to(device=device)

    """
    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
    
    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]
    model.load_state_dict(pretrained)
    """

    # loss_type = SSIMLoss().to(device=device)
    loss_type = MS_SSIM_L1_LOSS().to(device=device)  # 새로 만든 MS_SSIM_L1_LOSS 사용
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.RAdam(model.parameters(), args.lr)  # RAdam optimizer 사용


    # Check if a checkpoint exists, and load it if it does
    start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model, optimizer)


    # DataAugmentor 초기화
    current_epoch_fn = lambda: epoch
    augmentor = DataAugmentor(args, current_epoch_fn)

    # MaskAugmentor 초기화
    mask_augmentor = MaskAugmentor(current_epoch_fn, total_epochs=args.num_epochs)

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args
    )
    augmentor_arg = augmentor if augmentor.aug_on else None
    mask_augmentor_arg = mask_augmentor if args.mask_aug_on else None
    print(augmentor_arg)
    print(mask_augmentor_arg)
    train_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmentor=augmentor_arg,      # augmentor가 None이면 전달되지 않음
        mask_augmentor=mask_augmentor_arg  # mask_augmentor가 None이면 전달되지 않음
    )

    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")

    # 기존 val_loss_log 파일이 존재하면 불러오기, 없으면 빈 배열 생성
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))
    # print(val_loss_log)
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        p1 = augmentor.schedule_p()  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"MRAugmentation probability at epoch {epoch}: {p1}")
        randomacc = mask_augmentor.get_acc()  # 현재 epoch에 기반한 증강 확률을 계산
        p2 = mask_augmentor.maskAugProbability  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"mask_Augmentation probability at epoch {epoch}: {p2}")
        print(randomacc)
        

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, augmentor, mask_augmentor)
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
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
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

def initialize_model_and_optimizer(args, current_cascade_index, device, model_pt_filename, pre_model_pt_filename):
    model_path = args.exp_dir / model_pt_filename
    
    if model_path.exists():
        clear_gpu_memory()
        # model2.pt가 존재하는 경우
        print(f"Loading model2 from {model_path}")
        # PromptMR 모델을 사용하도록 변경
        model2 = PromptMR(
            num_cascades=args.second_cascade,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim = args.feature_dim,
            prompt_dim = args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim = args.sens_feature_dim,
            sens_prompt_dim = args.sens_prompt_dim,
            len_prompt = args.len_prompt,
            prompt_size = args.prompt_size,
            n_enc_cab = args.n_enc_cab,
            n_dec_cab = args.n_dec_cab,
            n_skip_cab = args.n_skip_cab,
            n_bottleneck_cab = args.n_bottleneck_cab,
            no_use_ca = args.no_use_ca,
            use_checkpoint=args.use_checkpoint,
            low_mem = args.low_mem
        )
        model2.to(device=device)

        # 첫 6개의 cascade block을 동결
        for i in range(args.pre_cascade):
            for param in model2.cascades[i].parameters():
                param.requires_grad = False  # 동결

        # Sensitivity map을 동결
        for param in model2.sens_net.parameters():
            param.requires_grad = False  # 동결

        # 마지막 3개의 cascade를 위한 별도의 optimizer 생성
        optimizers = [torch.optim.RAdam(filter(lambda p: p.requires_grad, model2.cascades[i].parameters()), args.lr) for i in range(args.pre_cascade, args.second_cascade)]

        # model2.pt 로드 (optimizer 포함)
        start_epoch, best_val_loss = load_checkpoint2(args.exp_dir, model2, optimizers, model_filename=model_pt_filename)
        clear_gpu_memory()

    else:
        # PromptMR 모델을 사용하도록 변경
        model1 = PromptMR(
            num_cascades=args.pre_cascade,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim = args.feature_dim,
            prompt_dim = args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim = args.sens_feature_dim,
            sens_prompt_dim = args.sens_prompt_dim,
            len_prompt = args.len_prompt,
            prompt_size = args.prompt_size,
            n_enc_cab = args.n_enc_cab,
            n_dec_cab = args.n_dec_cab,
            n_skip_cab = args.n_skip_cab,
            n_bottleneck_cab = args.n_bottleneck_cab,
            no_use_ca = args.no_use_ca,
            use_checkpoint=args.use_checkpoint,
            low_mem = args.low_mem
        )
        model1.to(device=device)

        # model1의 학습된 가중치를 불러옵니다.
        optimizer = torch.optim.RAdam(model1.parameters(), args.lr)

        # 마지막 3개의 cascade를 위한 별도의 optimizer 생성
        optimizers = [torch.optim.RAdam(filter(lambda p: p.requires_grad, model1.cascades[i].parameters()), args.lr) for i in range(args.pre_cascade-1, args.second_cascade-1)]

        
        if current_cascade_index == 0:
            start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model1, optimizer=optimizer, model_filename=pre_model_pt_filename)
        else :
            start_epoch, best_val_loss = load_checkpoint2(args.exp_dir, model1, optimizers=optimizers, model_filename=pre_model_pt_filename)

        # PromptMR 모델을 사용하도록 변경
        model2 = PromptMR(
            num_cascades=args.second_cascade,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim = args.feature_dim,
            prompt_dim = args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim = args.sens_feature_dim,
            sens_prompt_dim = args.sens_prompt_dim,
            len_prompt = args.len_prompt,
            prompt_size = args.prompt_size,
            n_enc_cab = args.n_enc_cab,
            n_dec_cab = args.n_dec_cab,
            n_skip_cab = args.n_skip_cab,
            n_bottleneck_cab = args.n_bottleneck_cab,
            no_use_ca = args.no_use_ca,
            use_checkpoint=args.use_checkpoint,
            low_mem = args.low_mem
        )
        model2.to(device=device)
        """
        # using pretrained parameter
        VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
        MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
        if not Path(MODEL_FNAMES).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

        pretrained = torch.load(MODEL_FNAMES)
        pretrained_copy = copy.deepcopy(pretrained)
        for layer in pretrained_copy.keys():
            if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
                del pretrained[layer]
        model.load_state_dict(pretrained)
        """

        # model1의 가중치를 model2로 복사합니다.
        # Sensitivity map 부분 복사 (동결하지 않고 학습 가능하도록 유지)
        model2.sens_net.load_state_dict(copy.deepcopy(model1.sens_net.state_dict()))

        # model1의 첫 6개의 cascade block을 복사하고 동결
        for i in range(args.pre_cascade):  # 첫 6개의 cascade block을 복사
            model2.cascades[i] = copy.deepcopy(model1.cascades[i])
            for param in model2.cascades[i].parameters():
                param.requires_grad = False  # 동결
        
        # Sensitivity map을 동결
        for param in model2.sens_net.parameters():
            param.requires_grad = False  # 동결

        # model1의 가중치를 복사한 후 GPU VRAM에서 model1 삭제하여 메모리 확보
        del model1
        clear_gpu_memory()

        # 마지막 3개의 cascade를 위한 별도의 optimizer 생성
        optimizers = [torch.optim.RAdam(filter(lambda p: p.requires_grad, model2.cascades[i].parameters()), args.lr) for i in range(args.pre_cascade, args.second_cascade)]

    return model2, optimizers, start_epoch, best_val_loss


def train2(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # val_loss_log 파일 로드
    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))

    if val_loss_log.size > 0:
        start_epoch = int(val_loss_log[-1, 0]) + 1
    else:
        start_epoch = 0
    current_cascade_index = 0
    for i, second_epoch in enumerate(args.second_epoch_list):
        if start_epoch < second_epoch:
            current_cascade_index = i
            args.num_epochs = second_epoch
            args.pre_cascade = args.cascade + i
            args.second_cascade = args.cascade + i+ 1
            break
    print("cascade개수 : ", args.cascade, args.pre_cascade, args.second_cascade, current_cascade_index)
    if args.second_cascade > 7:
        return

    # Cascade 개수를 반영한 모델 파일 이름 설정
    pre_model_pt_filename = f'model{args.pre_cascade:02d}.pt'
    model_pt_filename = f'model{args.second_cascade:02d}.pt'
    best_model_filename = f'best_model{args.second_cascade:02d}.pt'

    model2, optimizers, start_epoch, best_val_loss = initialize_model_and_optimizer(args, current_cascade_index, device, model_pt_filename, pre_model_pt_filename)

    # loss_type = SSIMLoss().to(device=device)
    loss_type = MS_SSIM_L1_LOSS().to(device=device)  # 새로 만든 MS_SSIM_L1_LOSS 사용

    # DataAugmentor 초기화
    current_epoch_fn = lambda: epoch
    augmentor = DataAugmentor(args, current_epoch_fn)

    # MaskAugmentor 초기화
    mask_augmentor = MaskAugmentor(current_epoch_fn, total_epochs=args.num_epochs)

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args
    )
    augmentor_arg = augmentor if augmentor.aug_on else None
    mask_augmentor_arg = mask_augmentor if args.mask_aug_on else None
    print(augmentor_arg)
    print(mask_augmentor_arg)
    train_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmentor=augmentor_arg,      # augmentor가 None이면 전달되지 않음
        mask_augmentor=mask_augmentor_arg  # mask_augmentor가 None이면 전달되지 않음
        # num_workers=4  # 여기에서 num_workers를 설정
    )

    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")

    # 기존 val_loss_log 파일이 존재하면 불러오기, 없으면 빈 배열 생성
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))
    # print(val_loss_log)

    # args.num_epochs = args.second_epoch
    for epoch in range(start_epoch, args.num_epochs):
        # Epoch와 second_epoch_list를 비교하여 cascade 조정
        if epoch >= args.second_epoch_list[current_cascade_index]:
            current_cascade_index += 1
            args.num_epochs = second_epoch
            args.pre_cascade = args.cascade + current_cascade_index
            args.second_cascade = args.cascade + current_cascade_index+ 1

            model_pt_filename = f'model{args.second_cascade:02d}.pt'
            pre_model_pt_filename = f'model{args.pre_cascade:02d}.pt'
            
            # Model 및 optimizer 초기화
            model2, optimizers, _, best_val_loss = initialize_model_and_optimizer(args, device, model_pt_filename, pre_model_pt_filename)


        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print("cascade개수 : ", args.cascade, args.pre_cascade, args.second_cascade, current_cascade_index)
        p1 = augmentor.schedule_p()  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"MRAugmentation probability at epoch {epoch}: {p1}")
        randomacc = mask_augmentor.get_acc()  # 현재 epoch에 기반한 증강 확률을 계산
        p2 = mask_augmentor.maskAugProbability  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"mask_Augmentation probability at epoch {epoch}: {p2}")
        print(randomacc)
        

        train_loss, train_time = train_epoch2(args, epoch, model2, train_loader, optimizers, loss_type, augmentor, mask_augmentor)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model2, val_loader)
        
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
        save_model2(args, args.exp_dir, epoch + 1, model2, optimizers, best_val_loss, is_new_best, model_filename=model_pt_filename)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )


def initialize_model_and_optimizer2(args, current_cascade_index, device, model_pt_filename, pre_model_pt_filename):
    model_path = args.exp_dir / model_pt_filename
    print(model_path)
    
    if model_path.exists():
        clear_gpu_memory()
        # model2.pt가 존재하는 경우
        print(f"Loading model2 from {model_path}")
        # PromptMR 모델을 사용하도록 변경
        # num_cascades=args.pre_cascade, 이게 맞음!!! 계속 additional_cascade_block이걸로 하나씩 올리는 것임!
        model2 = PromptMR2(
            num_cascades=args.pre_cascade,
            additional_cascade_block = args.additional_cascade_block,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim = args.feature_dim,
            prompt_dim = args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim = args.sens_feature_dim,
            sens_prompt_dim = args.sens_prompt_dim,
            len_prompt = args.len_prompt,
            prompt_size = args.prompt_size,
            n_enc_cab = args.n_enc_cab,
            n_dec_cab = args.n_dec_cab,
            n_skip_cab = args.n_skip_cab,
            n_bottleneck_cab = args.n_bottleneck_cab,
            no_use_ca = args.no_use_ca,
            use_checkpoint=args.use_checkpoint,
            low_mem = args.low_mem
        )
        model2.to(device=device)
        print(args.second_cascade)

        # 앞쪽 cascade block들을 동결
        for i in range(args.pre_cascade + args.additional_cascade_block * 6 - 6):
            for param in model2.cascades[i].parameters():
                param.requires_grad = False  # 동결

        # Sensitivity map 네트워크에서 앞쪽 블록들을 동결
        for i in range(args.additional_cascade_block):
            for param in model2.sens_nets[i].parameters():
                param.requires_grad = False  # 동결

        # 특정 파라미터들만 업데이트하도록 optimizer를 설정
        params_to_update = []
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 6].parameters())
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 5].parameters())
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 4].parameters())
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 3].parameters())
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 2].parameters())
        params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 1].parameters())

        optimizer = torch.optim.RAdam(params_to_update, args.lr)
        # model2.pt 로드 (optimizer 포함)
        print(model_pt_filename, args.second_cascade)
        start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model2, optimizer, model_filename=model_pt_filename)
        clear_gpu_memory()

    else:
        # 새로 학습할 모델 정의
        model2 = PromptMR2(
            num_cascades=args.pre_cascade,
            additional_cascade_block=args.additional_cascade_block,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim=args.feature_dim,
            prompt_dim=args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim=args.sens_feature_dim,
            sens_prompt_dim=args.sens_prompt_dim,
            len_prompt=args.len_prompt,
            prompt_size=args.prompt_size,
            n_enc_cab=args.n_enc_cab,
            n_dec_cab=args.n_dec_cab,
            n_skip_cab=args.n_skip_cab,
            n_bottleneck_cab=args.n_bottleneck_cab,
            no_use_ca=args.no_use_ca,
            use_checkpoint=args.use_checkpoint,
            low_mem=args.low_mem
        )
        model2.to(device=device)
        print(args.second_cascade)

        if args.additional_cascade_block >1:
            # 새 모델을 정의하고 이전 모델에서 가중치 불러오기
            model1 = PromptMR2(
                num_cascades=args.pre_cascade,
                additional_cascade_block=args.additional_cascade_block - 1,  # 이전 모델이므로 block을 줄여서 불러오기
                num_adj_slices=args.num_adj_slices,
                n_feat0=args.n_feat0,
                feature_dim=args.feature_dim,
                prompt_dim=args.prompt_dim,
                sens_n_feat0=args.sens_n_feat0,
                sens_feature_dim=args.sens_feature_dim,
                sens_prompt_dim=args.sens_prompt_dim,
                len_prompt=args.len_prompt,
                prompt_size=args.prompt_size,
                n_enc_cab=args.n_enc_cab,
                n_dec_cab=args.n_dec_cab,
                n_skip_cab=args.n_skip_cab,
                n_bottleneck_cab=args.n_bottleneck_cab,
                no_use_ca=args.no_use_ca,
                use_checkpoint=args.use_checkpoint,
                low_mem=args.low_mem
            )
            model1.to(device=device)

            optimizer = torch.optim.RAdam(model1.parameters(), args.lr)
            print(pre_model_pt_filename, "load_checkpoint1")
            start_epoch, best_val_loss = load_checkpoint(args.exp_dir, model1, optimizer=optimizer, model_filename=pre_model_pt_filename)

            # model1의 첫 6개의 cascade block을 복사하고 동결
            for i in range(args.pre_cascade + args.additional_cascade_block * 2 - 2):
                model2.cascades[i] = copy.deepcopy(model1.cascades[i])
                for param in model2.cascades[i].parameters():
                    param.requires_grad = False  # 동결
            
            # Sensitivity map 네트워크에서 앞쪽 블록들을 동결
            for i in range(args.additional_cascade_block):
                model2.sens_nets[i] = copy.deepcopy(model1.sens_nets[i])
                for param in model2.sens_nets[i].parameters():
                    param.requires_grad = False  # 동결
            model2.sens_nets[args.additional_cascade_block] = copy.deepcopy(model1.sens_nets[args.additional_cascade_block-1])

            # # 학습할 파라미터만 포함한 optimizer 생성
            # optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, model2.parameters()), args.lr)

            # 특정 파라미터들만 업데이트하도록 optimizer를 설정
            params_to_update = []
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 6].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 5].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 4].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 3].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 2].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 1].parameters())

            optimizer = torch.optim.RAdam(params_to_update, args.lr)


        else : 
            # PromptMR 모델을 사용하도록 변경
            model1 = PromptMR(
                num_cascades=args.pre_cascade,
                num_adj_slices=args.num_adj_slices,
                n_feat0=args.n_feat0,
                feature_dim = args.feature_dim,
                prompt_dim = args.prompt_dim,
                sens_n_feat0=args.sens_n_feat0,
                sens_feature_dim = args.sens_feature_dim,
                sens_prompt_dim = args.sens_prompt_dim,
                len_prompt = args.len_prompt,
                prompt_size = args.prompt_size,
                n_enc_cab = args.n_enc_cab,
                n_dec_cab = args.n_dec_cab,
                n_skip_cab = args.n_skip_cab,
                n_bottleneck_cab = args.n_bottleneck_cab,
                no_use_ca = args.no_use_ca,
                use_checkpoint=args.use_checkpoint,
                low_mem = args.low_mem
            )
            model1.to(device=device)

            # 마지막 3개의 cascade를 위한 별도의 optimizer 생성
            optimizers = [torch.optim.RAdam(filter(lambda p: p.requires_grad, model1.cascades[i].parameters()), args.lr) for i in range(args.pre_cascade-1, args.pre_cascade)]
            print(pre_model_pt_filename, "load_checkpoint2")
            start_epoch, best_val_loss = load_checkpoint2(args.exp_dir, model1, optimizers=optimizers, model_filename=pre_model_pt_filename)
            print("start_Epoch : ",start_epoch)
            # model1의 첫 6개의 cascade block을 복사하고 동결
            for i in range(args.pre_cascade + args.additional_cascade_block * 2 - 2):
                model2.cascades[i] = copy.deepcopy(model1.cascades[i])
                for param in model2.cascades[i].parameters():
                    param.requires_grad = False  # 동결
            
            # Sensitivity map 네트워크에서 앞쪽 블록들을 동결
            for i in range(args.additional_cascade_block):
                model2.sens_nets[i] = copy.deepcopy(model1.sens_net)
                for param in model2.sens_nets[i].parameters():
                    param.requires_grad = False  # 동결
            model2.sens_nets[args.additional_cascade_block] = copy.deepcopy(model1.sens_net)

            # # 학습할 파라미터만 포함한 optimizer 생성
            # optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, model2.parameters()), args.lr)

            # 특정 파라미터들만 업데이트하도록 optimizer를 설정
            params_to_update = []
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 6].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 5].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 4].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 3].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 2].parameters())
            params_to_update += list(model2.cascades[args.pre_cascade + args.additional_cascade_block * 6 - 1].parameters())

            optimizer = torch.optim.RAdam(params_to_update, args.lr)

        # model1의 가중치를 복사한 후 GPU VRAM에서 model1 삭제하여 메모리 확보
        del model1
        clear_gpu_memory()

    return model2, optimizer, start_epoch, best_val_loss

def train3(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # val_loss_log 파일 로드
    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))

    if val_loss_log.size > 0:
        start_epoch = int(val_loss_log[-1, 0]) + 1
    else:
        start_epoch = 0
    current_cascade_index = 0
    args.pre_cascade = 7
    for i, second_epoch in enumerate(args.second_epoch_list):
        if start_epoch < second_epoch:
            current_cascade_index = i
            args.num_epochs = second_epoch
            args.additional_cascade_block = i
            args.second_cascade = args.pre_cascade + args.additional_cascade_block*6
            break
    print("cascade개수 : ",args.additional_cascade_block, args.cascade, args.pre_cascade, args.second_cascade, current_cascade_index)

    # Cascade 개수를 반영한 모델 파일 이름 설정
    pre_model_pt_filename = f'model{args.second_cascade-6:02d}.pt'
    model_pt_filename = f'model{args.second_cascade:02d}.pt'
    best_model_filename = f'best_model{args.second_cascade:02d}.pt'

    model2, optimizer, start_epoch, best_val_loss = initialize_model_and_optimizer2(args, current_cascade_index, device, model_pt_filename, pre_model_pt_filename)

    # loss_type = SSIMLoss().to(device=device)
    loss_type = MS_SSIM_L1_LOSS().to(device=device)  # 새로 만든 MS_SSIM_L1_LOSS 사용

    # DataAugmentor 초기화
    current_epoch_fn = lambda: epoch
    augmentor = DataAugmentor(args, current_epoch_fn)

    # MaskAugmentor 초기화
    mask_augmentor = MaskAugmentor(current_epoch_fn, total_epochs=args.num_epochs)

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args
    )
    augmentor_arg = augmentor if augmentor.aug_on else None
    mask_augmentor_arg = mask_augmentor if args.mask_aug_on else None
    print(augmentor_arg)
    print(mask_augmentor_arg)
    train_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmentor=augmentor_arg,      # augmentor가 None이면 전달되지 않음
        mask_augmentor=mask_augmentor_arg  # mask_augmentor가 None이면 전달되지 않음
        # num_workers=4  # 여기에서 num_workers를 설정
    )

    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")

    # 기존 val_loss_log 파일이 존재하면 불러오기, 없으면 빈 배열 생성
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))
    print(val_loss_log)

    # args.num_epochs = args.second_epoch
    for epoch in range(start_epoch, args.num_epochs):
        # Epoch와 second_epoch_list를 비교하여 cascade 조정
        if epoch >= args.second_epoch_list[current_cascade_index]:
            current_cascade_index += 1
            args.num_epochs = second_epoch
            args.num_epochs = args.second_epoch_list[current_cascade_index]
            args.additional_cascade_block = current_cascade_index
            args.second_cascade = args.pre_cascade + args.additional_cascade_block*6

            model_pt_filename = f'model{args.second_cascade:02d}.pt'
            pre_model_pt_filename = f'model{args.second_cascade-2:02d}.pt'
            
            # Model 및 optimizer 초기화
            model2, optimizer, _, best_val_loss = initialize_model_and_optimizer2(args, device, model_pt_filename, pre_model_pt_filename)


        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        print("cascade개수 : ", args.cascade, args.pre_cascade, args.second_cascade, current_cascade_index)
        print(len(model2.cascades))
        p1 = augmentor.schedule_p()  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"MRAugmentation probability at epoch {epoch}: {p1}")
        randomacc = mask_augmentor.get_acc()  # 현재 epoch에 기반한 증강 확률을 계산
        p2 = mask_augmentor.maskAugProbability  # 현재 epoch에 기반한 증강 확률을 계산
        print(f"mask_Augmentation probability at epoch {epoch}: {p2}")
        print(randomacc)
        print("random acc range : ",mask_augmentor.later_acc_range)
        

        # train_loss, train_time = train_epoch(args, epoch, model2, train_loader, optimizer, loss_type, augmentor, mask_augmentor)
        train_loss, train_time = train_epoch3(args, epoch, model2, train_loader, optimizer, loss_type, augmentor, mask_augmentor)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model2, val_loader)
        
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
        save_model(args, args.exp_dir, epoch + 1, model2, optimizer, best_val_loss, is_new_best, model_filename=model_pt_filename)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
