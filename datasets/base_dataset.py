from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    def __init__(self, opt):
        self.crop_size = 512
        self.debug_mode = opt.debug_mode
        self.gamma = opt.gamma
        self.resolution = opt.resolution

    def norm_img(self, img, max_value):
        img = img / float(max_value)
        return img

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        # RGBG
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
    
    def center_crop(self, img, crop_size=None):
        H = img.shape[0]
        W = img.shape[1]

        if crop_size is not None:
            th, tw = crop_size[0], crop_size[1]
        else:
            th, tw = self.crop_size, self.crop_size
        x1_img = int(round((W - tw) / 2.))
        y1_img = int(round((H - th) / 2.))
        if img.ndim == 3:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw, :]
        else:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw]

        return input_patch

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
