#!/bin/bash

device=$1
export CUDA_VISIBLE_DEVICES=$device
choice=cls
frames=128
visual_dir=$PATH_TO_A_TSV    # one line in tsv like this:  video_id,frame_1,frame_2,...,frame_n
output_dir=$OUTPUT/VIT_${choice}_max${frames}frames/
mkdir -p $output_dir

python3 extract_by_shard.py \
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


