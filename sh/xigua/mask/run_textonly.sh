#!/bin/bash

device=0,1,2,3
export CUDA_VISIBLE_DEVICES=$device
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

cd /opt/tiger/fairseq_mmt
bash sh/xigua/mask/set_environment.sh


src_lang=en
tgt_lang=zh

mask=$1   #mask1,2,3,4,c,p
seed=$2
arch=$3
local_data_dir=/mnt/bd/xigua-data/fairseq_bin/xigua.en-zh.$mask.withtest

criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=7e-4
warmup=4000
max_tokens=4096
update_freq=1
patience=10
max_epoches=100
dropout=0.1

weight_decay=0.0
clip_norm=0.0


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

name=${mask}_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}_cn${clip_norm}_Realpatience${patience}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua/${mask}/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/xigua/${mask}
local_logs_dir=~/fairseq_logs/xigua/${mask}

hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put -f ${BASH_SOURCE[0]} $output_dir/train.sh




fairseq-train $local_data_dir \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num  -s $src_lang -t $tgt_lang \
  --arch $arch \
  --dropout $dropout \
  --weight-decay $weight_decay  \
  --criterion $criterion --label-smoothing 0.1 \
  --task translation \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm ${clip_norm}   \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq ${update_freq}  \
  --seed $seed \
  --patience $patience \
  --no-progress-bar  \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5,"lenpen":0.8}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --max-epoch ${max_epoches} --keep-interval-updates 10 \
  --no-epoch-checkpoints  \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi
