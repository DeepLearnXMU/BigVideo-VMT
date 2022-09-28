#!/bin/bash

device=$1
export CUDA_VISIBLE_DEVICES=$device
choice=$2
frames=$3
visual_dir=/mnt/bn/luyang/luyang/kangliyan/data/xigua/frames_tsv/128/
output_dir=/mnt/bn/luyang/kangliyan/data/xigua/VIT_${choice}_max${frames}frames/
mkdir -p $output_dir
python3 ~/fairseq_mmt/scripts/video_extractor/vit/extrach_by_shard.py \
    --max-num-frames $frames \
    --split $4 \
    --visual-dir $visual_dir    \
    --output-dir $output_dir \
    --video-feat-dim 768   \
    --img-res 224    \
    --video-feat-type vit_cls   \
    --choice $choice  \
    --shard_num $5 \
    --shard_id  $6

# -m torch.distributed.launch --nproc_per_node=2
