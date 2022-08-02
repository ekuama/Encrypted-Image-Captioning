import json
import os
import random

import h5py
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class ToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __call__(self, image, mask):
        return TF.to_tensor(image), TF.to_tensor(mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class HorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class VerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class Compose(tt.Compose):
    def __call__(self, image):
        for t in self.transforms:
            image[0], image[1] = t(image[0], image[1])
        return image[0], image[1]


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, aug):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.p = h5py.File(os.path.join(data_folder, 'phase_' + self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.p_imgs = self.p['images']
        self.a = h5py.File(os.path.join(data_folder, 'amp_' + self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.a_imgs = self.a['images']
        self.aug = aug

        # Captions per image
        self.cpi = self.p.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as c:
            self.captions = json.load(c)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as c:
            self.caplens = json.load(c)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform1 = tt.Compose([tt.Normalize((0.4975, 0.4975, 0.4975), (0.2897, 0.2897, 0.2896))])
        self.transform2 = tt.Compose([tt.Normalize((0.1378, 0.1314, 0.1211), (0.0722, 0.0688, 0.0634))])
        self.transforms = Compose([ToTensor(), HorizontalFlip(), VerticalFlip()])
        self.transform3 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4975, 0.4975, 0.4975), (0.2897, 0.2897, 0.2896))])
        self.transform4 = tt.Compose([tt.ToTensor(), tt.Normalize((0.1378, 0.1314, 0.1211), (0.0722, 0.0688, 0.0634))])

        # Total number of data points
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        p_img = self.p_imgs[i // self.cpi]
        a_img = self.a_imgs[i // self.cpi]
        if self.split == 'TEST':
            phase_img = self.transform3(p_img)
            amp_img = self.transform4(a_img)
        else:
            if self.aug:
                data = [p_img, a_img]
                p_img, a_img = self.transforms(data)
                phase_img = self.transform1(p_img)
                amp_img = self.transform2(a_img)
            else:
                phase_img = self.transform3(p_img)
                amp_img = self.transform4(a_img)

        caption = torch.LongTensor(self.captions[i])

        cap_len = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return phase_img, amp_img, caption, cap_len
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return phase_img, amp_img, caption, cap_len, all_captions

    def __len__(self):
        return self.dataset_size
