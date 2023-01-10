#!/bin/bash

device=$1
export CUDA_VISIBLE_DEVICES=$device
choice=$2
frames=$3
# visual_dir=/mnt/bn/luyang/kangliyan/data/vatex/frame_tsv_1211/128/
visual_dir=/mnt/bn/luyang/kangliyan/data/how2/frame_tsv_1209/128/
output_dir=/mnt/bd/xigua-youtube-3/how2/video_features/VIT_${choice}_max${frames}frames/
mkdir -p $output_dir

python3 /opt/tiger/fairseq_mmt/scripts/video_extractor/vit/extrach_by_shard.py \
    --max-num-frames ${frames} \
    --split $4 \
    --visual-dir $visual_dir    \
    --output-dir $output_dir \
    --video-feat-dim 768   \
    --img-res 224    \
    --choice $choice  \
    --shard_num $5 \
    --shard_id  $6 \
    --tsv_index 0

# python3 /opt/tiger/fairseq_mmt/scripts/video_extractor/vit/extrach_by_shard.py \
#     --max-num-frames 12 \
#     --split $4 \
#     --visual-dir $visual_dir    \
#     --output-dir $output_dir \
#     --video-feat-dim 768   \
#     --img-res 224    \
#     --choice $choice  \
#     --shard_num $5 \
#     --shard_id  $6 \
#     --tsv_index 1

# python3 /opt/tiger/fairseq_mmt/scripts/video_extractor/vit/extrach_by_shard.py \
#     --max-num-frames 12 \
#     --split $4 \
#     --visual-dir $visual_dir    \
#     --output-dir $output_dir \
#     --video-feat-dim 768   \
#     --img-res 224    \
#     --choice $choice  \
#     --shard_num $5 \
#     --shard_id  $6 \
#     --tsv_index 2

# python3 /opt/tiger/fairseq_mmt/scripts/video_extractor/vit/extrach_by_shard.py \
#     --max-num-frames 12 \
#     --split $4 \
#     --visual-dir $visual_dir    \
#     --output-dir $output_dir \
#     --video-feat-dim 768   \
#     --img-res 224    \
#     --choice $choice  \
#     --shard_num $5 \
#     --shard_id  $6 \
#     --tsv_index 3

# -m torch.distributed.launch --nproc_per_node=2
