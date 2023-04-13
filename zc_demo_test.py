import torch
import os
import cv2
import json
import numpy as np


# reso=16
# path = os.path.join("training-runs/wysiwyg/wysiwyg_8bit", reso_path_map[reso], "best_model.pkl")

# from training.networks_stylegan3_resetting import InvISPNet
from model.model import InvISPNet

if __name__ == '__main__':
    isp_path = 'invertibleISP_8b.pth'
    isp = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    isp.load_state_dict(torch.load(isp_path))
    
    print(isp.state_dict()['operations.0.F.conv1.weight'][0, 0])
    exit()
    
    isp.train(False)

    # raw = cv2.imread("data/wysiwyg/raw/1024x1024/build138_Telephoto_iPhone12Pro_Marietta_67/NXT_20211117_010919_713/1.png", -1)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    
    paths = []
    for _root, _dirs, _files in os.walk("test_imgs"):
        for _file in _files:
            path = os.path.join(_root, _file)
            paths.append(path)


    for idx in range(len(paths)):
        raw_ori =cv2.imread(paths[idx], -1)
        raw = cv2.cvtColor(raw_ori, cv2.COLOR_BGR2RGB)
        raw = raw.astype(np.float32)
        raw = torch.tensor(raw)
        
        raw = raw.permute(2, 0, 1)
        raw = raw[None, ...]
        
        # with open ("/sensei-fs/users/zhiwenc/datasets/wysiwyg_v1_meta.json") as f:
        #     cwb_dict = json.load(f)
        # cwb = cwb_dict["build138_Telephoto_iPhone12Pro_Marietta_67/NXT_20211117_010919_713/1"]['cwb'][:-1]
        # cwb = torch.tensor(cwb)
        # cwb = cwb / cwb.max()

        # raw = raw * cwb[..., None, None]
        raw = torch.pow(raw / 65535, 1/2.2) # Apply Gamma -> [0, 1]

        rgb = isp(raw) # any range
        rgb = torch.clamp(rgb, 0, 1)
        rgb = rgb.permute(0, 2, 3, 1)
        rgb = (rgb.detach().numpy() * 255).astype(np.uint8)
        rgb = np.squeeze(rgb)
        print(rgb.shape)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"test_imgs/test_{idx}_ori.png", raw_ori)
        


