import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders_
# from utils.model.varnet import VarNet
from promptMR.models.promptmr import PromptMR

import os

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (koutput, grappa, iinput, _, _, fnames, slices) in data_loader:
            koutput = kspace.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            iinput = iinput.cuda(non_blocking=True)
            input_combined = torch.cat((koutput, grappa, iinput), dim=1)
            output = model(input_combined)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # model = VarNet(num_cascades=args.cascade, 
    #                chans=args.chans, 
    #                sens_chans=args.sens_chans)
    # model.to(device=device)


    # val_loss_log 파일 로드
    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log3.npy")
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
            # args.num_epochs = second_epoch
            args.pre_cascade = args.cascade + i
            args.second_cascade = args.cascade + i+ 1
            break
    print(start_epoch)
    if start_epoch<args.num_epochs:
        args.second_cascade = args.cascade
    print("cascade개수 : ", args.cascade, args.pre_cascade, args.second_cascade, current_cascade_index)

    model_pt_filename = 'model_naf.pt'
    model = NAFNet(
        img_channel=args.img_channel_naf,
        width=args.width_naf,
        middle_blk_num=args.middle_blk_num_naf,
        enc_blk_nums=args.enc_blk_nums_naf,
        dec_blk_nums=args.dec_blk_nums_naf
    )
    model.to(device=device)

    print(model_pt_filename)
    checkpoint = torch.load(args.exp_dir / model_pt_filename, map_location='cpu')
    # checkpoint = torch.load(args.exp_dir / best_model_filename, map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders_naf(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)