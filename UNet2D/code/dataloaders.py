import os
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset


def random_flip(data, p=0.5):
    dims = len(data.shape)
    if dims == 2:
        data = np.expand_dims(data, axis=0)

    if random.random() < p:
        data = np.flip(data, axis=1)  
    if random.random() < p:
        data = np.flip(data, axis=2)

    if dims == 2:
        return data[0]
    return data


def random_rot90(data, p=0.5):
    dims = len(data.shape)
    if dims == 2:
        data = np.expand_dims(data, axis=0)

    if random.random() < p:
        k = np.random.choice((1, 2, 3))
        data = np.rot90(data, k, axes=(1, 2))

    if dims == 2:
        return data[0]
    return data


def apply_intensity_augs(image,
                         brightness_delta=0.0,
                         contrast_range=(1.0, 1.0),
                         gamma_range=(1.0, 1.0),
                         noise_std=0.0):
    """
    Int transforms (image only no label)
    """
    img_dims = len(image.shape)
    if img_dims == 2:
        img = image.astype(np.float32)
    else:
        img = image[0].astype(np.float32) 

    if brightness_delta > 0:
        delta = np.random.uniform(-brightness_delta, +brightness_delta)
        img = img + delta

    cmin, cmax = contrast_range
    if cmax > 1.0 or cmin < 1.0:
        factor = np.random.uniform(cmin, cmax)
        mean = img.mean()
        img = (img - mean) * factor + mean

    gmin, gmax = gamma_range
    if gmax > 1.0 or gmin < 1.0:
        gamma = np.random.uniform(gmin, gmax)
        m, M = np.percentile(img, [0.5, 99.5])
        if M > m:
            img = np.clip((img - m) / (M - m), 0, 1)
            img = np.power(img, gamma)
            img = img * (M - m) + m

    if noise_std > 0:
        img = img + np.random.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    if img_dims == 2:
        return img
    else:
        return np.expand_dims(img, axis=0)


class ACDCTrainDataset(Dataset):
    def __init__(self,
                 base_dir,
                 flip=True,
                 rot=True,
                 brightness_delta=0.0,
                 contrast_range=(1.0, 1.0),
                 gamma_range=(1.0, 1.0),
                 noise_std=0.0):
        self._base_dir = base_dir
        self.sample_list = glob(os.path.join(base_dir, 'slice', '*.npy'))
        self.flip = flip
        self.rot = rot

        # intensity params
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = np.load(self.sample_list[idx])

        if self.flip:
            data = random_flip(data)
        if self.rot:
            data = random_rot90(data)

        image = data[0:1]
        label = data[1] 

        image = apply_intensity_augs(
            image,
            brightness_delta=self.brightness_delta,
            contrast_range=self.contrast_range,
            gamma_range=self.gamma_range,
            noise_std=self.noise_std
        )

       
        image = torch.from_numpy(image.copy()).float()    
        label = torch.from_numpy(label.copy()).long()   

        return image, label
