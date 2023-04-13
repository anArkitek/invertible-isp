
block_num=8
batch_size=24

torchrun --nnodes=1 --nproc_per_node=8\
        train.py \
        --task=${block_num}blocks \
        --raw_root=./data/wysiwyg/raw/1024x1024 \
        --rgb_root=./data/wysiwyg/img/1024x1024 \
        --meta_root=/sensei-fs/users/zhiwenc/datasets/wysiwyg_v1_meta.json \
        --gamma \
        --aug \
        --out_path=./exps/ \
        --resolution=512 \
        --block_num=${block_num} \
        --batch_size=${batch_size}


# CUDA_VISIBLE_DEVICES=0 python train.py \
#                 --task=debug \
#                 --data_path="./data/" \
#                 --gamma \
#                 --aug \
#                 --camera="wysiwyg" \
#                 --out_path="./exps/" \
#                 --resolution=256 \
#                 --block_num=8
                # --debug_mode

# python train.py --task=debug2 \
#                 --data_path="./data/" \
#                 --gamma \
#                 --aug \
#                 --camera="Canon_EOS_5D" \
#                 --out_path="./exps/" \
#                 --debug_mode
