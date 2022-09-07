#!/bin/bash

device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

cd /opt/tiger/fairseq_mmt
bash sh/xigua/mask/set_environment.sh

src_lang=en
tgt_lang=zh



criterion=cross_modal_criterion
if [ $criterion == "label_smoothed_cross_entropy" ]; then
        cri=LSCE
    elif [ $criterion == "cross_modal_criterion" ]; then
        cri=CMC
    elif [ $criterion == "cross_modal_criterion_with_ctr" ]; then
        cri=CMCCTR
fi

mask=$1   #mask1,2,3,4,c,p
seed=$2
arch=$3

local_data_dir=/mnt/bd/xigua-data/fairseq_bin/xigua.en-zh.$mask.withtest


fp16=1 #0
lr=7e-4
warmup=4000
max_tokens=4096
update_freq=1
max_updates=1500000
patience=10
dropout=0.1

weight_decay=0.1
clip_norm=0.0
residual_policy="None"
ini_alpha=0.0



video_feat_type="VIT_cls"
if [ $video_feat_type == "VIT_cls"  ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bd/xigua-data/features/VIT_cls/
        max_vid_len=32
        feature_choice=cls
  elif [ $video_feat_type == "VIT_patch_avg" ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bd/xigua-data/features/VIT_patch/
        max_vid_len=197
        feature_choice=patch
fi

max_num_frames=32
img_res=224
freeze_backbone=True

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`


name=${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_vtype${video_feat_type}_mf${max_num_frames}_rp${residual_policy}_ia${ini_alpha}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua/${mask}/ve_clip_double_cross_att/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/xigua/${mask}/ve_clip_double_cross_att
local_logs_dir=~/fairseq_logs/xigua/${mask}/ve_clip_double_cross_att

hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put -f ${BASH_SOURCE[0]} $output_dir/train.sh



fairseq-train $local_data_dir \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
  --arch $arch \
  --dropout $dropout \
  --weight-decay $weight_decay  \
  --clip-norm ${clip_norm}   \
  --criterion $criterion --label-smoothing 0.1 \
  --task raw_video_translation \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq  \
  --seed $seed \
  --no-progress-bar  \
  --find-unused-parameters \
  --log-interval 1 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5,"lenpen":0.8}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --patience $patience \
  --max-epoch ${max_epoches} --keep-interval-updates 10 \
  --visual-dir $visual_dir --video-feat-dim $video_feat_dim --img-res 224 --video-feat-type $video_feat_type   \
  --features-choice ${feature_choice} --freeze-backbone \
  --residual-policy $residual_policy --ini-alpha $ini_alpha \
  --num-workers 8 \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi