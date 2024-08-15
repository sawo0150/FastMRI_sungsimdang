import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet
from utils.learning.test_part_naf import forward
import time
    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
    # parser.add_argument('--cascade', type=int, default=20, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    # DataAugmentor 관련 argument 추가
    parser.add_argument('--aug-on', default=False, action='store_true', help='Switch to turn data augmentation on')
    

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
    parser.add_argument('--use_checkpoint',type=bool, default=False, help='Use gradient checkpointing to save memory')
    parser.add_argument('--low_mem',type=bool, default=True, help='Use low memory settings')

    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_loss_dir = '../result' / args.net_name

    args.num_epochs = 65
    args.second_epoch_list = [126, 250, 400, 500, 600]

    args.second_cascade = 10

    args.num_workers = 8

    args.image_channel_naf = 3
    args.width_naf = 32
    args.middle_blk_num_naf = 1
    args.enc_blk_nums_naf = [1,1,1,28]
    args.dec_blk_nums_naf = [1,1,1,1]
    args.num_epochs_naf = 20
    args.lr_naf = 1e-3
    
    public_acc, private_acc = None, None

    assert(len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
      if acc in ['acc4', 'acc5', 'acc8']:
        public_acc = acc
      else:
        private_acc = acc
        
    assert(None not in [public_acc, private_acc])
    
    start_time = time.time()
    
    # Public Acceleration
    args.data_path = args.path_data / public_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'public'
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    # Private Acceleration
    args.data_path = args.path_data / private_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'private'
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')
    
    print('Success!') if reconstructions_time < 3000 else print('Fail!')