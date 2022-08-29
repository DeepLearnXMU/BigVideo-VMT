#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CUDA_VISIBLE_DEVICES=1

src_lang=en
tgt_lang=zh



criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.001
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_epoches=100
dropout=0.3
seed=1207
weight_decay=0.1
clip_norm=0.0
arch=transformer_vatex
gpu_num=1
mask=mask_verb_35565  #mask1,2,3,4,c,p

local_data_dir=~/data/fairseq_bin_filter/vatex.en-zh.${mask}



name=textonly_filter_${mask}_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0809/$mask/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex_0809/$mask
local_logs_dir=~/fairseq_logs/vatex_0809/$mask

hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put -f ${BASH_SOURCE[0]} $output_dir/train.sh

fairseq-train $local_data_dir   \
  --save-dir $output_dir  \
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang  \
  --arch $arch  \
  --dropout $dropout  \
  --weight-decay $weight_decay  \
  --criterion $criterion --label-smoothing 0.1  \
  --task translation  \
  --optimizer adam --adam-betas '(0.9, 0.98)'  \
  --clip-norm ${clip_norm}  \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup  \
  --max-tokens $max_tokens --update-freq $update_freq --max-epoch $max_epoches  \
  --find-unused-parameters  \
  --seed $seed  \
  --no-progress-bar  \
  --eval-bleu  \
  --eval-bleu-args '{"beam": 5,"lenpen":0.8}'  \
  --eval-bleu-detok moses  \
  --eval-bleu-remove-bpe  \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  \
  --patience $patience  \
  --keep-last-epochs $keep_last_epochs   \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}



echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name}  $LOGS_DIR/log.${name}

