from __future__ import print_function, division
from genericpath import isfile
import os, random
import torch
import numpy as np
from torch.utils.data import Dataset
import rawpy
from glob import glob
# from scipy.imageio import imread
from imageio import imread
from .base_dataset import BaseDataset
import sys
import json
import cv2


class WYSDatasetTrain(Dataset):
    def __init__(self, opt):
        super().__init__(opt=opt) 
        self.patch_size = self.resolution
        self.raw_paths = self.load_paths(opt.raw_root)
        self.rgb_paths = self.load_paths(opt.rgb_root)
        self.meta_dic = self.load_json(opt.meta_root)

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        rgb_path = self.rgb_paths[idx]
        wb = self.meta_dic[idx]['cwb']
        
        input_raw_img = cv2.imread(raw_path, -1)
        target_rgb_img = cv2.imread(rgb_path, -1)
        wb = wb / wb.max() 
        input_raw_img = input_raw_img * wb[:-1]   

        self.patch_size = self.resolution
        input_raw_img, target_rgb_img = self.aug(self.patch_size, input_raw_img, target_rgb_img, flow=True, demos=True)  

        if self.gamma:
            norm_value = np.power(65535, 1/2.2)
            input_raw_img = np.power(input_raw_img, 1/2.2)
        else:
            norm_value = 4095 if self.camera_name=='Canon_EOS_5D' else 16383

        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img).float()
        target_rgb_img = self.np2tensor(target_rgb_img).float()
        target_raw_img = self.np2tensor(target_raw_img).float()
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'meta': {'raw_path': raw_path, 'rgb_path': rgb_path, 'wb': wb}}

    def load_paths(self, root):
        if os.path.isfile(root) and root.endswith('txt'):
            with open(root, 'r') as f:
                lines = f.readlines()
                paths = sorted([line[:-1] for line in lines])
                return paths
        else:
            sys.exit('NOT SUPPORTED PATH ROOT!')

    def load_json(self, root):
        with open (root, 'r') as f:
            data = json.load(f)
        return data

    def aug(self, patch_size, input_raw, target_rgb, flow=False, demos=False):
        input_raw, target_rgb = self.random_crop(patch_size, input_raw,target_rgb,flow=flow, demos=demos)
        input_raw, target_rgb = self.random_rotate(input_raw,target_rgb)
        input_raw, target_rgb = self.random_flip(input_raw,target_rgb)

    def np2tensor(self, nparr):
        return torch.Tensor(nparr.astype(np.float32)).permute(2,0,1)


class FiveKDatasetTrain(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt=opt) 
        self.patch_size = self.resolution
        input_RAWs_WBs, target_RGBs = self.load(is_train=True)
        assert len(input_RAWs_WBs) == len(target_RGBs)        
        self.data = {'input_RAWs_WBs':input_RAWs_WBs, 'target_RGBs':target_RGBs} 

    def random_flip(self, input_raw, target_rgb):
        idx = np.random.randint(2)
        input_raw = np.flip(input_raw,axis=idx).copy()
        target_rgb = np.flip(target_rgb,axis=idx).copy()
        
        return input_raw, target_rgb

    def random_rotate(self, input_raw, target_rgb):
        idx = np.random.randint(4)
        input_raw = np.rot90(input_raw,k=idx)
        target_rgb = np.rot90(target_rgb,k=idx)

        return input_raw, target_rgb

    def random_crop(self, patch_size, input_raw, target_rgb,flow=False,demos=False):
        H, W, _ = input_raw.shape
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))

        patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        if flow or demos:
            patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        else:
            patch_target_rgb = target_rgb[rnd_h*2:rnd_h*2 + patch_size*2, rnd_w*2:rnd_w*2 + patch_size*2, :]

        return patch_input_raw, patch_target_rgb
        
    def aug(self, patch_size, input_raw, target_rgb, flow=False, demos=False):
        input_raw, target_rgb = self.random_crop(patch_size, input_raw,target_rgb,flow=flow, demos=demos)
        input_raw, target_rgb = self.random_rotate(input_raw,target_rgb)
        input_raw, target_rgb = self.random_flip(input_raw,target_rgb)
        
        return input_raw, target_rgb

    def __len__(self):
        return len(self.data['input_RAWs_WBs'])

    def __getitem__(self, idx):    
        input_raw_wb_path = self.data['input_RAWs_WBs'][idx]
        target_rgb_path = self.data['target_RGBs'][idx]
        
        target_rgb_img = imread(target_rgb_path)
        input_raw_wb = np.load(input_raw_wb_path)
        input_raw_img = input_raw_wb['raw']
        wb = input_raw_wb['wb']
        wb = wb / wb.max() 
        input_raw_img = input_raw_img * wb[:-1]   

        self.patch_size = self.resolution
        input_raw_img, target_rgb_img = self.aug(self.patch_size, input_raw_img, target_rgb_img, flow=True, demos=True)  

        if self.gamma:            
            norm_value = np.power(4095, 1/2.2) if self.camera_name=='Canon_EOS_5D' else np.power(16383, 1/2.2)            
            input_raw_img = np.power(input_raw_img, 1/2.2)
        else:
            norm_value = 4095 if self.camera_name=='Canon_EOS_5D' else 16383

        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img).float()
        target_rgb_img = self.np2tensor(target_rgb_img).float()
        target_raw_img = self.np2tensor(target_raw_img).float()
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'file_name':input_raw_wb_path.split("/")[-1].split(".")[0]}
        return sample

class FiveKDatasetTest(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt=opt)
        self.patch_size = self.resolution
        
        input_RAWs_WBs, target_RGBs = self.load(is_train=False)
        assert len(input_RAWs_WBs) == len(target_RGBs)        
        self.data = {'input_RAWs_WBs':input_RAWs_WBs, 'target_RGBs':target_RGBs} 

    def __len__(self):
        return len(self.data['input_RAWs_WBs'])

    def __getitem__(self, idx):    
        input_raw_wb_path = self.data['input_RAWs_WBs'][idx]
        target_rgb_path = self.data['target_RGBs'][idx]
        
        target_rgb_img = imread(target_rgb_path)
        input_raw_wb = np.load(input_raw_wb_path)
        input_raw_img = input_raw_wb['raw']
        wb = input_raw_wb['wb']
        wb = wb / wb.max() 
        input_raw_img = input_raw_img * wb[:-1]

        if self.gamma:            
            norm_value = np.power(4095, 1/2.2) if self.camera_name=='Canon_EOS_5D' else np.power(16383, 1/2.2)            
            input_raw_img = np.power(input_raw_img, 1/2.2)             
        else:
            norm_value = 4095 if self.camera_name=='Canon_EOS_5D' else 16383

        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img).float()
        target_rgb_img = self.np2tensor(target_rgb_img).float()
        target_raw_img = self.np2tensor(target_raw_img).float()
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'file_name':input_raw_wb_path.split("/")[-1].split(".")[0]}
        return sample

