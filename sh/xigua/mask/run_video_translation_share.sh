#!/bin/bash

device=0,1,2,3
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


mask=${1}  #mask1,2,3,4,c,p
seed=${2}
arch=${3}
video_feat_type=${4}
weight_decay=${5}
lr=${6}
warmup=${7}
max_tokens=${8}
dropout=${9}
video_dropout=${10}
max_vid_len=${11}
text_data=${12}
id_type=${13}


if [ ${text_data} == "original_share" ]; then
    local_data_dir=/mnt/bn/luyang/kangliyan/data/fairseq_bin/xigua.en-zh.annotations1016_share/
    elif [ ${text_data} == "asr_share" ]; then
      local_data_dir=~/data/fairseq_bin/annotations_1016_asr_1109_share
    elif [ ${text_data} == "action_label_top1_zh" ]; then
      local_data_dir=/mnt/bn/luyang/kangliyan/data/fairseq_bin/xigua.en-zh.annotations1016.action_label_top1_zh
    elif [ ${text_data} == "action_label_upto0.5_zh" ]; then
      local_data_dir=/mnt/bn/luyang/kangliyan/data/fairseq_bin/xigua.en-zh.annotations1016.action_label_upto0.5_zh

fi


fp16=1 #0
update_freq=1
max_epoches=100
patience=10


clip_norm=0.0




video_ids_path=/mnt/bd/xigua-data/text/preprocessd_v1

if [ $video_feat_type == "VIT_cls"  ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bd/xigua-data/features/VIT_cls/
  elif  [ $video_feat_type == "VIT_128" ]; then
        video_feat_dim=768
#        video_feat_path=/mnt/bn/luyang/kangliyan/data/xigua/VIT_cls_max128frames/
        video_feat_path=/mnt/bd/image-128frames-tsv/VIT_cls_max128frames/

  elif [ $video_feat_type == "VIT_patch_avg" ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bd/xigua-data/features/VIT_patch/
  elif [ $video_feat_type == "slowfast" ]; then
        video_feat_dim=2304
        video_feat_path=/mnt/bd/xigua-slowfast-videoswin/slowfast/features/slowfast/
  elif [ $video_feat_type == "videoswin" ]; then
        video_feat_dim=1024
        video_feat_path=/mnt/bd/xigua-slowfast-videoswin/videoswin/

fi



gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`


name=${mask}ed20_${text_data}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_mt${max_tokens}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_dp${dropout}_vtype${video_feat_type}_mvlen${max_vid_len}_vdp${video_dropout}_idtype${id_type}_patience${patience}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua/${mask}/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/xigua/${mask}/
local_logs_dir=~/fairseq_logs/xigua/${mask}/


hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put -f ${BASH_SOURCE[0]} $output_dir/train.sh



fairseq-train $local_data_dir \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
  --arch $arch --share-all-embeddings \
  --dropout $dropout \
  --weight-decay $weight_decay  \
  --clip-norm ${clip_norm}   \
  --criterion $criterion --label-smoothing 0.1    \
  --task raw_video_translation_from_np \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq  \
  --seed $seed \
  --no-progress-bar  \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5,"lenpen":0.8}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --max-epoch ${max_epoches} --keep-interval-updates 10  --keep-best-checkpoints 10   \
  --patience $patience \
  --video-feat-path $video_feat_path \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim \
  --video-feat-type $video_feat_type \
  --max-vid-len $max_vid_len   \
  --video-dropout $video_dropout  \
  --id-type $id_type  \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi