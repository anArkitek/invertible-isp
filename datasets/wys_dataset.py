import os
import torch
import numpy as np
from .base_dataset import BaseDataset
import sys
import json
import cv2
import random


class WYSDatasetTrain(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt) 
        self.opt = opt
        self.patch_size = self.resolution
        self.raw_paths = self.load_paths(opt.raw_root)
        # self.raw_paths = [path for path in self.raw_paths if path.endswith('2.png')]
        self.rgb_paths = self.load_paths(opt.rgb_root)
        # self.rgb_paths = [path for path in self.rgb_paths if path.endswith('2.png')]
        self.meta_dic = self.load_json(opt.meta_root)

        num_for_training = int(len(self.raw_paths) * 0.99)
        self.raw_paths = self.raw_paths[ : num_for_training]
        self.rgb_paths = self.rgb_paths[ : num_for_training]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        rgb_path = self.rgb_paths[idx]
        # print(raw_path)
        # print(rgb_path)
        # print('=' * 100)
        rel_path = os.path.relpath(raw_path, self.opt.raw_root)
        rel_path = rel_path[:-4]
        wb = np.array(self.meta_dic[rel_path]['cwb'])
        
        input_raw_img = cv2.imread(raw_path, -1)
        target_rgb_img = cv2.imread(rgb_path, -1)
        input_raw_img = cv2.cvtColor(input_raw_img, cv2.COLOR_BGR2RGB)
        target_rgb_img = cv2.cvtColor(target_rgb_img, cv2.COLOR_BGR2RGB)
        assert input_raw_img.shape == target_rgb_img.shape
        # wb = wb / wb.max()
        # input_raw_img = input_raw_img * wb[:-1]

        self.patch_size = self.resolution
        input_raw_img, target_rgb_img = self.aug(self.patch_size, input_raw_img, target_rgb_img, flow=True, demos=True)
        
        input_raw_img = input_raw_img / 65535
        target_rgb_img = target_rgb_img / 255

        # if self.gamma:
        #     norm_value = np.power(65535, 1/2.2)
        #     input_raw_img = np.power(input_raw_img, 1/2.2)
        # else:
        #     norm_value = 65535

        # target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        # input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img)
        target_rgb_img = self.np2tensor(target_rgb_img)
        target_raw_img = self.np2tensor(target_raw_img)
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'meta': {'raw_path': raw_path, 'rgb_path': rgb_path, 'wb': wb}}
        return sample

    def load_paths(self, root):
        if os.path.isfile(root) and root.endswith('txt'):
            with open(root, 'r') as f:
                lines = f.readlines()
                paths = sorted([line[:-1] for line in lines])
                return paths
        elif os.path.isdir(root):
            paths = []
            for _root, _dirs, _files in os.walk(root):
                for _file in _files:
                    paths.append(os.path.join(_root, _file))
            paths = sorted(paths)
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
        return input_raw, target_rgb

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
        rnd_h = random.randint(0, max(0, (H - patch_size) // 2))
        rnd_w = random.randint(0, max(0, (W - patch_size) // 2))
        assert input_raw.shape == target_rgb.shape
        patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        if flow or demos:
            patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        else:
            patch_target_rgb = target_rgb[rnd_h*2:rnd_h*2 + patch_size*2, rnd_w*2:rnd_w*2 + patch_size*2, :]

        # print(f'H: {H} || W: {W} || rnd_h: {rnd_h} || rnd_w: {rnd_w} || input_raw: {input_raw.shape} || target_rgb: {target_rgb.shape}\
        #  || patch_input_raw: {patch_input_raw.shape} || patch_target_rgb: {patch_target_rgb.shape}')

        return patch_input_raw, patch_target_rgb

    def np2tensor(self, nparr):
        return torch.from_numpy(nparr.astype(np.float32)).permute(2,0,1)

###############################################################################

class WYSDatasetTest(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt.raw_root 
        self.patch_size = self.resolution
        self.raw_paths = self.load_paths(opt.raw_root)
        # self.raw_paths = [path for path in self.raw_paths if path.endswith('2.png')]
        self.rgb_paths = self.load_paths(opt.rgb_root)
        # self.rgb_paths = [path for path in self.rgb_paths if path.endswith('2.png')]
        self.meta_dic = self.load_json(opt.meta_root)

        num_for_training = int(len(self.raw_paths) * 0.99)
        self.raw_paths = self.raw_paths[num_for_training: ]
        self.rgb_paths = self.rgb_paths[num_for_training: ]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        rgb_path = self.rgb_paths[idx]
        rel_path = os.path.relpath(raw_path, self.opt.raw_root)
        rel_path = rel_path[:-4]

        wb = np.array(self.meta_dic[rel_path]['cwb'])
        
        input_raw_img = cv2.imread(raw_path, -1)
        target_rgb_img = cv2.imread(rgb_path, -1)
        input_raw_img = cv2.cvtColor(input_raw_img, cv2.COLOR_BGR2RGB)
        target_rgb_img = cv2.cvtColor(target_rgb_img, cv2.COLOR_BGR2RGB)
        wb = wb / wb.max()
        input_raw_img = input_raw_img * wb[:-1]

        if self.gamma:
            norm_value = np.power(65535, 1/2.2)
            input_raw_img = np.power(input_raw_img, 1/2.2)
        else:
            norm_value = 65535

        self.patch_size = self.resolution
        # input_raw_img, target_rgb_img = self.aug(self.patch_size, input_raw_img, target_rgb_img, flow=True, demos=True)  
        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img)
        target_rgb_img = self.np2tensor(target_rgb_img)
        target_raw_img = self.np2tensor(target_raw_img)
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'meta': {'raw_path': raw_path, 'rgb_path': rgb_path, 'wb': wb}}
        return sample

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

    def np2tensor(self, nparr):
        return torch.from_numpy(nparr.astype(np.float32)).permute(2,0,1)