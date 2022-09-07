#!/bin/bash

device=$1
export CUDA_VISIBLE_DEVICES=$device

choice=$2
visual_dir=/mnt/bd/xigua-data/tsv/
output_dir=/mnt/bd/xigua-data/features/VIT_${choice}/
mkdir -p $output_dir
python3 /opt/tiger/fairseq_mmt/sh/vatex/mask/extract_shard.py \
    --max-num-frames 32 \
    --split $2 \
    --visual-dir $visual_dir    \
    --output-dir $output_dir \
    --video-feat-dim 768   \
    --img-res 224    \
    --video-feat-type vit_cls   \
    --choice $choice  \
    --shard_num $3 \
    --shard_id  $4

# -m torch.distributed.launch --nproc_per_node=2
