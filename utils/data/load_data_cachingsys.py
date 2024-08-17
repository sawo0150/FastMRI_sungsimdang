import h5py
import random
from utils.data.transforms import DataTransform_cachingsys
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from collections import deque

class SliceData_cache(Dataset):
    def __init__(self, root, max_key,transform, input_key, target_key,cache_size=50, augmentor=None, mask_augmentor=None, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.max_key = max_key
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

        self.cache_size = cache_size
        self.cache = deque()  # Cache without fixed size     

        # Create a list of unique .h5 files instead of slice examples
        self.available_files = list(set(kspace_file for kspace_file, _ in self.kspace_examples))
        self.image_file_map = {str(kspace_fname): str(image_fname) for (image_fname, _), (kspace_fname, _) in zip(self.image_examples, self.kspace_examples)} if not forward else {}

    def reset_available_files(self):
        # Reset the available files at the beginning of each epoch
        self.available_files = list(set(kspace_file for kspace_file, _ in self.kspace_examples))
        print("new epoch start! Reset available files...")

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def _load_from_cache(self):
        if len(self.cache) == 0:
            return None
        random_index = random.randint(0, len(self.cache) - 1)
        data = self.cache[random_index]
        del self.cache[random_index]  # Remove the used item from the cache
        return data

    def _add_to_cache(self, kspace_fname, image_fname, input_data, mask_data, target_data, maximum):
        for input, mask, target, max_val in zip(input_data, mask_data, target_data, maximum):
            self.cache.append((input, mask, target, max_val))

    def _fill_cache(self):
        while len(self.cache) < self.cache_size and self.available_files:
            kspace_fname = self.available_files.pop(random.randint(0, len(self.available_files) - 1))

            with h5py.File(kspace_fname, "r") as hf:
                input_data = hf[self.input_key][:]
                mask = np.array(hf["mask"])

            if not self.forward:
                image_fname = self.image_file_map[str(kspace_fname)]
                with h5py.File(image_fname, "r") as hf:
                    target_data = hf[self.target_key][:]
                    attrs = dict(hf.attrs)
                    maximum = [attrs[self.max_key]] * len(input_data)
                    mask_data = [mask] * len(input_data)
            else:
                target_data = [-1] * len(input_data)
                maximum = [-1] * len(input_data)
                mask_data = [mask] * len(input_data)

            # Add each slice as an individual element in the cache
            self._add_to_cache(kspace_fname, image_fname, input_data, mask_data, target_data, maximum)

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        # if not self.forward:
        #     image_fname, _ = self.image_examples[i]
        # kspace_fname, dataslice = self.kspace_examples[i]

        # with h5py.File(kspace_fname, "r") as hf:
        #     input = hf[self.input_key][dataslice]
        #     mask =  np.array(hf["mask"])
        # if self.forward:
        #     target = -1
        #     attrs = -1
        # else:
        #     with h5py.File(image_fname, "r") as hf:
        #         target = hf[self.target_key][dataslice]
        #         attrs = dict(hf.attrs)
        
        self._fill_cache()
        data = self._load_from_cache()

        input, mask, target, maximum = data

        # MaskAugmentor가 있을 경우 mask에 적용
        if self.mask_augmentor:
            mask = self.mask_augmentor.augment(mask)
            # print(mask)
        # print(maximum)
        # print("transform 전",mask.shape, input.shape)
        mask, kspace, target = self.transform(mask, input, target)
    
        # print("transform 후",mask.shape, kspace.shape)

        # Augmentor가 있을 경우 kspace와 target에 적용
        if self.augmentor:
            input, target = self.augmentor(kspace, target, target_size=target.shape[-2:])
        
        return mask, kspace, target, maximum
    
def create_data_loaders_cache(data_path, args, shuffle=False, isforward=False, augmentor=None, mask_augmentor=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    data_storage = SliceData_cache(
        root=data_path,
        max_key = max_key_,
        transform=DataTransform_cachingsys(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        augmentor=augmentor,  # augmentor를 전달
        mask_augmentor=mask_augmentor,  # mask_augmentor를 전달
        forward=isforward
    )

    # Hook to reset available files at the start of each epoch
    def on_epoch_start_hook(loader):
        loader.dataset.reset_available_files()

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    # Attach the epoch start hook to the loader
    data_loader.on_epoch_start = lambda: on_epoch_start_hook(data_loader)

    return data_loader
