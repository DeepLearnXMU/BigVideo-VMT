
#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128




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

mask=mask_verb_3    #mask1,2,3,4,c,p
if [ $mask == "mask0" ]; then
        local_data_dir=~/data/en_zh.char
else
        local_data_dir=~/data/vatex/fairseq_bin/vatex.en-zh.${mask}
fi
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
arch=vatex_double_crossatt_pewln

video_feat_path=~/data/vatex/video/images_resized/vit_base_patch16_224
video_ids_path=~/data/raw_texts/ids
video_feat_type="I3D"
if [ $video_feat_type == "VIT_cls"  ]; then
        video_feat_dim=768
        video_feat_path=~/data/vatex/video/images_resized/vit_base_patch16_224
  elif [ $video_feat_type == "VIT_patch" ]; then
        video_feat_dim=768
        video_feat_path=~/data/vatex/video/images_resized/vit_base_patch16_224
  elif [ $video_feat_type == "I3D" ]; then
        video_feat_dim=1024
        video_feat_path=~/data/vatex_features/
fi


gpu_num=1


name=vatex_${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_vtype${video_feat_type}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0731/${mask}/doubleCA/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex_0731/${mask}/doubleCA
local_logs_dir=~/fairseq_logs/vatex_0731/${mask}/doubleCA

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
  --criterion $criterion --label-smoothing 0.1 --report-modal-similarity \
  --task vatex_translation \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq --max-epoch $max_epoches \
  --log-interval 1 \
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
  --max-vid-len 15   \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $output_dir/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $output_dir/log.${name}