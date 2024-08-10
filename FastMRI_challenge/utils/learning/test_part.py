import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
# from utils.model.varnet import VarNet
from promptMR.models.promptmr import PromptMR

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

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

    
    # model2.pt 파일의 존재 여부를 확인
    model2_path = args.exp_dir / 'model2.pt'

    if model2_path.exists():
        model = PromptMR(
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
        model.to(device=device)

        
        checkpoint = torch.load(args.exp_dir / 'model2.pt', map_location='cpu')
        # checkpoint = torch.load(args.exp_dir / 'best_model2.pt', map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        model.load_state_dict(checkpoint['model'])
        
        forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
        reconstructions, inputs = test(args, model, forward_loader)
        save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
    else:
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

        
        checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        model.load_state_dict(checkpoint['model'])
        
        forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
        reconstructions, inputs = test(args, model, forward_loader)
        save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)