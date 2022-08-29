
#!/bin/bash

device=0,1,2,3
export CUDA_VISIBLE_DEVICES=$device
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

mask=mask0    #mask1,2,3,4,c,p
local_data_dir=~/data/fairseq_bin_filter/vatex.en-zh.${mask}


fp16=1 #0
lr=0.001
warmup=4000
max_tokens=1024
update_freq=1
keep_last_epochs=10
patience=10
max_epoches=100
dropout=0.3
seed=1207
weight_decay=0.1
clip_norm=0.0
residual_policy="None"
ini_alpha=0.0
arch=ve_double_cross_att_small


video_feat_type="videoswin"
if [ $video_feat_type == "clip" ]; then
        video_feat_dim=512
        visual_dir=/mnt/bn/luyang/data/vatex/images_resized_r3/
  elif [ $video_feat_type == "videoswin" ]; then
        videoswin_size=base
        kinetics=600
        video_feat_dim=1024
        videoswin_path=/root/models/swin_base_patch244_window877_kinetics600_22k.pth
        visual_dir=/mnt/bn/luyang/data/vatex/images_resized_r3/
fi
max_num_frames=32
img_res=224
freeze_backbone=True

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`


name=vatex_${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_vtype${video_feat_type}_mf${max_num_frames}_rp${residual_policy}_ia${ini_alpha}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0829/${mask}/ve_double_cross_att/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex_0829/${mask}/ve_double_cross_att
local_logs_dir=~/fairseq_logs/vatex_0829/${mask}/ve_double_cross_att

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
  --max-tokens $max_tokens --update-freq $update_freq --max-epoch $max_epoches \
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
  --keep-last-epochs $keep_last_epochs  \
  --visual-dir $visual_dir --video-feat-dim $video_feat_dim --img-res 224 --video-feat-type $video_feat_type   \
  --vidswin-size $videoswin_size --kinetics $kinetics --videoswin-path ${videoswin_path} --grid-feat --freeze-backbone \
  --residual-policy $residual_policy --ini-alpha $ini_alpha \
  --num-workers 8 \
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

put_result=$?
if [ $put_result == 1  ]; then
        hdfs dfs -put -f $local_logs_dir/log.${name} $LOGS_DIR/log.${name}
fi