import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
# from utils.model.varnet import VarNet
from promptMR.models.promptmr import PromptMR2
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet

import os

def test(args, model, model_naf, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, grappa, iinput, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            iinput = iinput.cuda(non_blocking=True)
            koutput = model(kspace, mask)
            
            koutput = koutput.unsqueeze(1)
            grappa = grappa.unsqueeze(1)
            iinput = iinput.unsqueeze(1)
            input_combined = torch.cat([koutput, grappa, iinput], dim=1)
            output = model_naf(koutput)
            output = output.squeeze(1)

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
    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log_naf.npy")
    if os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
    else:
        val_loss_log = np.empty((0, 2))

    if val_loss_log.size > 0:
        start_epoch = int(val_loss_log[-1, 0]) + 1
    else:
        start_epoch = 0
        

    model = PromptMR2(
        num_cascades=args.pre_cascade,
        additional_cascade_block = args.additional_cascade_block,
        # additional_cascade_block = 3,
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

    print(args.exp_dir / "best_model13.pt")
    checkpoint = torch.load(args.exp_dir / "best_model13.pt", map_location='cpu')
    # checkpoint = torch.load(args.exp_dir / "best_model13.pt", map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])

    model_naf = NAFNet(
        img_channel=args.img_channel_naf,
        width=args.width_naf,
        middle_blk_num=args.middle_blk_num_naf,
        enc_blk_nums=args.enc_blk_nums_naf,
        dec_blk_nums=args.dec_blk_nums_naf
    )
    model_naf.to(device=device)

    checkpoint = torch.load(args.exp_dir / "model_naf.pt", map_location='cpu')
    model_naf.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders_naf(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, model_naf, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)