#!/bin/bash

device=0
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
weight_decay=$4
lr=$5
warmup=$6
max_tokens=$7
dropout=$8
text_data=${9}
patience=${10}

if [ $text_data == "original" ]; then
    local_data_dir=~/data/fairseq_bin/vatex.en-zh.bpe15k
    elif [ ${text_data} == "fromxigua+youtube+vatex" ]; then
      local_data_dir=/mnt/bd/xigua-youtube-2/data/fairseq_bin/vatex.en-zh.fromxigua+youtube+vatex
fi


criterion=label_smoothed_cross_entropy
fp16=1 #0


update_freq=1
max_epoches=100



clip_norm=0.0


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

name=${mask}ed20_${text_data}_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}_cn${clip_norm}_dp${dropout}_Realpatience${patience}_length256

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex/finetune/${mask}/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex/finetune/${mask}
local_logs_dir=~/fairseq_logs/vatex/finetune/${mask}

hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put -f ${BASH_SOURCE[0]} $output_dir/train.sh




fairseq-train $local_data_dir \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num  -s $src_lang -t $tgt_lang \
  --arch $arch  --max-source-positions 256 --max-target-positions 256 \
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
  --skip-invalid-size-inputs-valid-test  \
  --eval-bleu \
  --eval-bleu-args '{"beam": 4,"lenpen":1.0}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --max-epoch ${max_epoches} --keep-last-epochs 10 --keep-best-checkpoints 10  \
  --no-epoch-checkpoints  \
  --finetune-from-model hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua+youtube+wmt19/mask0/mask0ed20_fromxigua+youtube+vatex_archtransformer_tgtzh_lr7e-4_wu4000_seed1207_gpu8_mt4096_acc1_wd0.1_cn0.0_dp0.3_Realpatience10_length256/checkpoint_best.pt \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}


echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi
