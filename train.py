import numpy as np
import os, time, random
import argparse
import json

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model.model import InvISPNet
from datasets.wys_dataset import WYSDatasetTrain
from config.config import get_arguments

from utils.JPEG import DiffJPEG

from torch.utils.tensorboard import SummaryWriter


# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.system('rm tmp')

# DiffJPEG = DiffJPEG(differentiable=True, quality=90).cuda()


def main():
    # torch.autograd.set_detect_anomaly(True)
    args = get_arguments()
    
    # keep track of whether the current process is the `master` process
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    batch_size_per_gpu = args.batch_size // args.world_size
    assert batch_size_per_gpu * args.world_size == args.batch_size
    if args.rank == 0:
        print("Parsed arguments: {}".format(args))
        os.makedirs(args.out_path, exist_ok=True)
        os.makedirs(args.out_path + "%s"%args.task, exist_ok=True)
        os.makedirs(args.out_path + "%s/checkpoint"%args.task, exist_ok=True)

        with open(args.out_path + "%s/commandline_args.yaml"%args.task , 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    print(f"Start running basic DDP example on rank {args.rank}.")    

    net = InvISPNet(channel_in=3, channel_out=3, block_num=args.block_num).to(args.local_rank)
    ddp_net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
    
    RAWDataset = WYSDatasetTrain(opt=args)        
    sampler = DistributedSampler(RAWDataset,shuffle=True)
    dataloader = DataLoader(dataset=RAWDataset, batch_size=batch_size_per_gpu, sampler=sampler)
    
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)

    if args.rank == 0:
        writer = SummaryWriter(f'runs/{args.task}')

    if args.rank == 0:
        print("==================== Start to train ====================")

    step = 0
    for epoch in range(0, 300):
        epoch_time = time.time()             
        total_loss = 0.
        dataloader.sampler.set_epoch(epoch)
        dist.barrier()
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time() 

            input, target_rgb, target_raw = sample_batched['input_raw'].cuda(), sample_batched['target_rgb'].cuda(), \
                                        sample_batched['target_raw'].cuda()
            
            reconstruct_rgb = ddp_net(input)
            reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)
            rgb_loss = F.l1_loss(reconstruct_rgb, target_rgb)

            # zhiwen: we do not use jpeg for now
            # reconstruct_rgb = DiffJPEG(reconstruct_rgb)
            
            #####################
            reconstruct_raw = ddp_net(reconstruct_rgb, rev=True)
            raw_loss = F.l1_loss(reconstruct_raw, target_raw)
            loss = args.rgb_weight * rgb_loss + raw_loss
            #####################
            
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.rank == 0:
                # print("task: %s Epoch: %d Step: %d || loss: %.5f raw_loss: %.5f rgb_loss: %.5f || lr: %f time: %f"%(
                #     args.task, epoch, step, loss.detach().cpu().numpy(), raw_loss.detach().cpu().numpy(), 
                #     rgb_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
                # )) 
                print("task: %s Epoch: %d Step: %d || loss: %.5f raw_loss: %.5f rgb_loss: %.5f || lr: %f time: %f"%(
                    args.task, epoch, step, loss.detach().cpu().numpy(), raw_loss.detach().cpu().numpy(), 
                    rgb_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
                )) 
            step += 1 
        
        if args.rank == 0:
            writer.add_scalar('train_loss', total_loss / len(RAWDataset), epoch)
            torch.save(ddp_net.state_dict(), args.out_path+"%s/checkpoint/latest.pth"%args.task)
        dist.barrier()
        
        if (epoch + 1) % 10 == 0 and args.rank == 0:
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(ddp_net.state_dict(), args.out_path+"%s/checkpoint/%04d.pth"%(args.task, epoch))
            print("[INFO] Successfully saved "+args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
        dist.barrier()
        scheduler.step()
        
        if args.rank == 0:
            print("[INFO] Epoch time: ", time.time()-epoch_time, "task: ", args.task)

    
    print(f"exit rank: {args.rank}")

if __name__ == '__main__':
    # torch.set_num_threads(4)
    main()
