#!/bin/bash


export CUDA_VISIBLE_DEVICES=4
src_lang=en
tgt_lang=zh

criterion=label_smoothed_cross_entropy
if [ $criterion == "label_smoothed_cross_entropy" ]; then
        cri=LSCE
    elif [ $criterion == "cross_modal_criterion" ]; then
        cri=CMC
    elif [ $criterion == "cross_modal_criterion_with_ctr" ]; then
        cri=CMCCTR
fi


mask=mask_verb_35565    #mask1,2,3,4,c,p
local_data_dir=~/data/fairseq_bin_filter/vatex.en-zh.${mask}

#data=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh

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
arch=vatex_multimodal_transformer_att_vatex_top_pe


video_feat_path=~/data/vatex/video/images_resized_r3/vit_base_patch16_224
video_ids_path=~/data/vatex/raw_texts/filter_ids/
video_feat_type="I3D"
if [ $video_feat_type == "VIT_cls"  ]; then
        video_feat_dim=768
        video_feat_path=~/data/vatex/video/images_resized_r3/vit_base_patch16_224
        max_vid_len=15
  elif [ $video_feat_type == "VIT_patch_avg" ]; then
        video_feat_dim=768
        video_feat_path=~/data/vatex/video/images_resized_r3/vit_base_patch16_224
        max_vid_len=200
  elif [ $video_feat_type == "I3D" ]; then
        video_feat_dim=1024
        video_feat_path=~/data/vatex_features/
        max_vid_len=32
fi

SA_attention_dropout=0.1
SA_video_dropout=0.1


gpu_num=1


name=vatex_${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}_cn${clip_norm}_patience${patience}_avdp${SA_video_dropout}_aadp${SA_attention_dropout}_vtype${video_feat_type}_vlen${max_vid_len}
output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0809/${mask}/sa/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex_0809/${mask}/sa
local_logs_dir=~/fairseq_logs/vatex_0809/${mask}/sa

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
  --task vatex_translation \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq --max-epoch $max_epoches \
  --find-unused-parameters \
  --seed $seed \
  --no-progress-bar  \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5,"lenpen":0.8}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --patience $patience \
  --keep-last-epochs $keep_last_epochs  \
  --video-feat-path $video_feat_path \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim \
  --video-feat-type $video_feat_type \
  --max-vid-len $max_vid_len   \
  --SA-video-dropout ${SA_video_dropout} --SA-attention-dropout ${SA_attention_dropout} \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi

