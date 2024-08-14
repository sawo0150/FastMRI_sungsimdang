import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
from torch.utils.data._utils.collate import default_collate  # default_collate 함수 임포트

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, augmentor=None, mask_augmentor=None, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.augmentor = augmentor  # augmentor를 인자로 받음
        self.mask_augmentor = mask_augmentor  # mask_augmentor를 인자로 받음
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
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
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        # MaskAugmentor가 있을 경우 mask에 적용
        if self.mask_augmentor:
            mask = self.mask_augmentor.augment(mask)
        
        # print("transform 전",mask.shape, input.shape)
        mask, kspace, target, maximum, fname, slice = self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
    
        # print("transform 후",mask.shape, kspace.shape)

        # Augmentor가 있을 경우 kspace와 target에 적용
        if self.augmentor:
            input, target = self.augmentor(kspace, target, target_size=target.shape[-2:])
        
        return mask, kspace, target, maximum, fname, slice
    
def create_data_loaders(data_path, args, shuffle=False, isforward=False, augmentor=None, mask_augmentor=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        augmentor=augmentor,  # augmentor를 전달
        mask_augmentor=mask_augmentor,  # mask_augmentor를 전달
        forward=isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,  # num_workers 설정
        pin_memory=True  # GPU 메모리에 데이터를 고정하여 더 빠르게 전송
    )
    return data_loader

class SizeGroupedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self._group_by_size()

    def _group_by_size(self):
        groups = defaultdict(list)
        for idx, (fname, _) in enumerate(self.data_source.kspace_examples):
            with h5py.File(fname, "r") as hf:
                shape = hf[self.data_source.input_key].shape[1:3]  # height, width
            groups[shape].append(idx)
        return groups

    def __iter__(self):
        batches = []
        for group in self.groups.values():
            random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batch = group[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
                elif not self.drop_last and len(batch) > 0:
                    batches.append(batch)  # 마지막 남은 데이터들을 하나의 배치로 추가
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(g) // self.batch_size for g in self.groups.values())
        else:
            return sum((len(g) + self.batch_size - 1) // self.batch_size for g in self.groups.values())

def create_data_loaders2(data_path, args, shuffle=False, isforward=False, augmentor=None, mask_augmentor=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        augmentor=augmentor,  # augmentor를 전달
        mask_augmentor=mask_augmentor,  # mask_augmentor를 전달
        forward=isforward
    )

    batch_sampler = SizeGroupedBatchSampler(data_storage, args.batch_size)

    data_loader = DataLoader(
        dataset=data_storage,
        batch_sampler=batch_sampler,
        collate_fn=lambda x: default_collate(x)  # 기본 collate_fn 사용
    )
    return data_loader