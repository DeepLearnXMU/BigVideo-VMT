#!/bin/bash

checkpoint_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex/textonly_char_archtransformer_vatex_tgtzh_lr0.001_wu2000_me100_seed1207_gpu1_mt4096_acc1_wd0.1_cn0.0_patience10
local_output_dir=~/fairseq_results
mkdir -p $local_output_dir

script_root=/opt/tiger/fairseq_mmt/perl
multi_bleu=$script_root/multi-bleu.perl
who=valid
test_DATA=~/data/en_zh.char
ensemble=3
video_feat_path=~/data/vatex_features
video_ids_path=~/data/raw_texts/ids
video_feat_dim=1024




checkpoint=checkpoint_best.pt
length_penalty=0.8
best_point=46


if [ -n "$ensemble" ]; then
      if [ ! -e "$checkpoint_dir/last$ensemble.ensemble.pt" ]; then
              PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $checkpoint_dir --output $checkpoint_dir/last$ensemble.up$best_point.pt --num-epoch-checkpoints $ensemble  \
                                                                      --checkpoint-upper-bound $best_point
      fi
      checkpoint=last$ensemble.up$best_point.pt
fi


echo "-----$who ensemble-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size  128  \
--lenpen $length_penalty \
--task vatex_translation \
--video-feat-path $video_feat_path \
--video-ids-path $video_ids_path \
--video-feat-dim $video_feat_dim \
--output $local_output_dir/$local_output_dir.$length_penalty.gen-$who.log | tee $local_output_dir/$checkpoint.$length_penalty.gen-$who.log

echo "move to $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log "
hdfs dfs -put $local_output_dir/$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log