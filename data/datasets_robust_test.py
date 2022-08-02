import json
import os

import h5py
import torch
import torchvision.transforms as tt
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, name):
        self.name = name

        # Open hdf5 file where images are stored
        self.p = h5py.File(os.path.join(data_folder, self.name + '_phase_TEST_IMAGES_' + data_name + '.hdf5'), 'r')
        self.p_imgs = self.p['images']
        self.a = h5py.File(os.path.join(data_folder, self.name + '_amp_TEST_IMAGES_' + data_name + '.hdf5'), 'r')
        self.a_imgs = self.a['images']

        # Captions per image
        self.cpi = self.p.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, 'TEST_CAPTIONS_' + data_name + '.json'), 'r') as c:
            self.captions = json.load(c)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, 'TEST_CAPLENS_' + data_name + '.json'), 'r') as c:
            self.caplens = json.load(c)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        if self.name == 'noise_0.25':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4975, 0.4975, 0.4975),
                                                                      (0.2897, 0.2897, 0.2896))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.1258, 0.1199, 0.1105),
                                                                      (0.0665, 0.0634, 0.0584))])
        elif self.name == 'noise_0.5':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4975, 0.4975, 0.4975),
                                                                      (0.2897, 0.2897, 0.2896))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.1176, 0.1121, 0.1033),
                                                                      (0.0634, 0.0605, 0.0557))])
        elif self.name == 'noise_1':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4975, 0.4975, 0.4975),
                                                                      (0.2897, 0.2897, 0.2896))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.1071, 0.1021, 0.0941),
                                                                      (0.0607, 0.0579, 0.0533))])
        elif self.name == 'exclude_0.2':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4980, 0.4980, 0.4980),
                                                                      (0.2590, 0.2590, 0.2590))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.1102, 0.1051, 0.0968),
                                                                      (0.0848, 0.0809, 0.0745))])
        elif self.name == 'exclude_0.4':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4979, 0.4980, 0.4980),
                                                                      (0.2247, 0.2247, 0.2246))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.0828, 0.0789, 0.0727),
                                                                      (0.0828, 0.0789, 0.0727))])
        elif self.name == 'exclude_0.6':
            self.transform1 = tt.Compose([tt.ToTensor(), tt.Normalize((0.4985, 0.4985, 0.4986),
                                                                      (0.1830, 0.1830, 0.1830))])
            self.transform2 = tt.Compose([tt.ToTensor(), tt.Normalize((0.0597, 0.0581, 0.0541),
                                                                      (0.0882, 0.0859, 0.0800))])

        # Total number of data points
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        p_img = self.p_imgs[i // self.cpi]
        a_img = self.a_imgs[i // self.cpi]
        phase_img = self.transform1(p_img)
        amp_img = self.transform2(a_img)

        caption = torch.LongTensor(self.captions[i])

        cap_len = torch.LongTensor([self.caplens[i]])

        all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        return phase_img, amp_img, caption, cap_len, all_captions

    def __len__(self):
        return self.dataset_size
