#!/bin/bash

visual_dir=/mnt/bn/luyang/kangliyan/data/xigua/images/
output_dir=/mnt/bn/luyang/kangliyan/data/xigua/videoswin/
mkdir -p $output_dir
python3 extract.py \
    --max-num-frames 32 \
    --split $1 \
    --visual-dir $visual_dir    \
    --output-dir $output_dir \
    --video-feat-dim 1024   \
    --img-res 224    \
    --vidswin-size base --kinetics 600   \
    --videoswin-path ~/models/swin_base_patch244_window877_kinetics600_22k.pth  \
    --shard_num $2 \
    --shard_id  $3


