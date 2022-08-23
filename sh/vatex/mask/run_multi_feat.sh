
#!/bin/bash


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

mask=mask35565 #mask1,2,3,4,c,p
local_data_dir=~/data/fairseq_bin_filter/vatex.en-zh.${mask}


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
arch=vatex_double_crossatt_multi_pewln

residual_policy="None"
ini_alpha=0.0


video_feat_path=~/data/vatex/images_resized_r3/vit_base_patch16_224
video_ids_path=~/data/vatex/raw_texts/filter_ids/
video_feat_type1="VIT_cls"
video_feat_type2="slowfast"

if [ $video_feat_type1 == "VIT_cls"  ]; then
        video_feat_dim1=768
        video_feat_path1=~/data/vatex/images_resized_r3/vit_base_patch16_224
        max_vid_len1=32
  elif [ $video_feat_type1 == "VIT_patch_avg" ]; then
        video_feat_dim1=768
        video_feat_path1=~/data/vatex/images_resized_r3/vit_base_patch16_224
        max_vid_len1=197
  elif [ $video_feat_type1 == "I3D" ]; then
        video_feat_dim1=1024
        video_feat_path1=~/data/vatex_features/
        max_vid_len1=32
  elif [ $video_feat_type1 == "slowfast" ]; then
        video_feat_dim1=2304
        video_feat_path1=~/data/vatex/slowfast/
        max_vid_len1=36
  elif [ $video_feat_type1 == "slowfast13" ]; then
        video_feat_dim1=2304
        video_feat_path1=~/data/vatex/slowfast13/
        max_vid_len1=54
  elif [ $video_feat_type1 == "clip_vit16" ]; then
        video_feat_dim1=512
        video_feat_path1=~/data/vatex/clip_vit16/
        max_vid_len1=54
  elif [ $video_feat_type1 == "videoswin" ]; then
        video_feat_dim1=1024
        video_feat_path1=~/data/vatex/videoswin/
        max_vid_len1=12
fi

if [ $video_feat_type2 == "VIT_cls"  ]; then
        video_feat_dim2=768
        video_feat_path2=~/data/vatex/images_resized_r3/vit_base_patch16_224
        max_vid_len2=32
  elif [ $video_feat_type2 == "VIT_patch_avg" ]; then
        video_feat_dim2=768
        video_feat_path2=~/data/vatex/images_resized_r3/vit_base_patch16_224
        max_vid_len2=197
  elif [ $video_feat_type2 == "I3D" ]; then
        video_feat_dim2=1024
        video_feat_path2=~/data/vatex_features/
        max_vid_len2=32
  elif [ $video_feat_type2 == "slowfast" ]; then
        video_feat_dim2=2304
        video_feat_path2=~/data/vatex/slowfast/
        max_vid_len2=36
  elif [ $video_feat_type2 == "slowfast13" ]; then
        video_feat_dim2=2304
        video_feat_path2=~/data/vatex/slowfast13/
        max_vid_len2=54
  elif [ $video_feat_type2 == "clip_vit16" ]; then
        video_feat_dim2=512
        video_feat_path2=~/data/vatex/clip_vit16/
        max_vid_len1=54
  elif [ $video_feat_type2 == "videoswin" ]; then
        video_feat_dim2=1024
        video_feat_path2=~/data/vatex/videoswin/
        max_vid_len2=12
fi


gpu_num=1


name=vatex_${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_cn${clip_norm}_vtype1${video_feat_type1}_vtype2${video_feat_type2}_vlen${max_vid_len}_rp${residual_policy}_ia${ini_alpha}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0823/${mask}/doubleCA/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex_0823/${mask}/doubleCA
local_logs_dir=~/fairseq_logs/vatex_0823/${mask}/doubleCA

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
  --criterion $criterion --label-smoothing 0.1 --report-modal-similarity --report-layer-alpha  \
  --task vatex_translation_multi_feats \
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
  --video-feat-path $video_feat_path1 $video_feat_path2 \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim1  $video_feat_dim2 \
  --video-feat-type $video_feat_type1 $video_feat_type2 \
  --max-vid-len $max_vid_len1 $max_vid_len2   \
  --residual-policy $residual_policy --ini-alpha $ini_alpha \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $output_dir/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi