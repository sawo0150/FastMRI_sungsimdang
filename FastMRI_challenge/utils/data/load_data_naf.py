import h5py
import random
import torch
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
from torch.utils.data._utils.collate import default_collate  # default_collate 함수 임포트
from promptMR.models.promptmr import PromptMR

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, augmentor=None, mask_augmentor=None, forward=False, model=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        #self.augmentor = augmentor  # augmentor를 인자로 받음
        #self.mask_augmentor = mask_augmentor  # mask_augmentor를 인자로 받음
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.model = model
        
        
        image_files = list(Path(root / "image").iterdir())
        for fname in sorted(image_files):
            num_slices = self._get_metadata(fname)

            self.image_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        
        image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            kinput = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        with h5py.File(image_fname, "r") as hf:
            grappa = hf["image_grappa"][dataslice]
            iinput = hf["image_input"][dataslice]
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        # # MaskAugmentor가 있을 경우 mask에 적용
        # if self.mask_augmentor:
        #     mask = self.mask_augmentor.augment(mask)
        
        # print("transform 전",mask.shape, input.shape)
        mask, kspace, target, maximum, fname, slice = self.transform(mask, kinput, target, attrs, kspace_fname.name, dataslice)
        grappa = to_tensor(grappa)
        iinput = to_tensor(iinput)
        # mask, kspace를 model에 넣은 결과를 알아야 한다
        
        
        self.model.eval()
        with torch.no_grad():
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            koutput = self.model(kspace, mask)
        # # print("transform 후",mask.shape, kspace.shape)

        # # Augmentor가 있을 경우 kspace와 target에 적용
        # if self.augmentor:
        #     input, target = self.augmentor(kspace, target, target_size=target.shape[-2:])
        
        return koutput, grappa, iinput, target, maximum, fname, slice
    
def create_data_loaders_naf(data_path, args, shuffle=False, isforward=False, augmentor=None, mask_augmentor=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    model_pt_filename = f'model{args.second_cascade:02d}.pt'
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
    checkpoint = torch.load(args.exp_dir / model_pt_filename, map_location='cpu')
    # checkpoint = torch.load(args.exp_dir / best_model_filename, map_location='cpu')

    model.load_state_dict(checkpoint['model'])
        
    
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        augmentor=augmentor,  # augmentor를 전달
        mask_augmentor=mask_augmentor,  # mask_augmentor를 전달
        forward=isforward,
        model=model
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,  # num_workers 설정
        pin_memory=True  # GPU 메모리에 데이터를 고정하여 더 빠르게 전송
    )
    return data_loader