python test_rgb.py \
            --task=test_rgb_8blocks \
            --raw_root=/sensei-fs/users/zhiwenc/datasets/wysiwyg_v1_dngval_stage3png_16bit_originalSize.txt \
            --rgb_root=/sensei-fs/users/zhiwenc/datasets/wysiwyg_v1_dngval_finalpng_8bit_originalSize.txt \
            --meta_root=/sensei-fs/users/zhiwenc/datasets/wysiwyg_v1_meta.json \
            --gamma \
            --out_path=./exps/ \
            --resolution=256 \
            --block_num=8 \
            --ckpt=./exps/8blocks/checkpoint/0099.pth

# python test_rgb.py --task=pretrained \
#                 --data_path="./data/" \
#                 --gamma \
#                 --camera="NIKON_D700" \
#                 --out_path="./exps/" \
#                 --ckpt="./exps/8blocks/checkpoint/latest.pth" \
#                 # --split_to_patch

# python test_raw.py --task=pretrained \
#                 --data_path="./data/" \
#                 --gamma \
#                 --camera="NIKON_D700" \
#                 --out_path="./exps/" \
#                 --ckpt="./pretrained/nikon.pth" \
#                 --split_to_patch

