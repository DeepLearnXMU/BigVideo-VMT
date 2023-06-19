#!/bin/bash

device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device




src_lang=en
tgt_lang=zh



criterion=cross_modal_criterion_with_ctr_revise
cri=CMCCTRRE


seed=1
arch=video_fushion_encoder_revise_one_merge_before_pewln
video_feat_type=ViT
weight_decay=0.1
lr=7e-4
warmup=4000
max_tokens=4096
update_freq=1
dropout=0.1
video_dropout=0.0
max_vid_len=12
train_sampling_strategy=uniform
patience=10
contrastive_strategy=mean+mlp
contrastive_weight=1.0
contrastive_temperature=0.002


local_data_dir=$BIN_DIR




fp16=1 #0
max_epoches=100
patience=10

clip_norm=0.0

video_ids_path=$DATA_DIR
if  [ $video_feat_type == "ViT" ]; then
        video_feat_dim=768
        video_feat_path=$DATA_DIR/video_features/ViT
  elif [ $video_feat_type == "slowfast" ]; then
        video_feat_dim=2304
        video_feat_path=$DATA_DIR/video_features/SlowFast
fi



gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`


name=${mask}_arch${arch}_cri${cri}_tgt${tgt_lang}_lr${lr}_wu${warmup}_mat${max_tokens}_acc${update_freq}_me${max_epoches}_seed${seed}_gpu${gpu_num}_wd${weight_decay}_dp${dropout}_vtype${video_feat_type}_mvlen${max_vid_len}_ts${train_sampling_strategy}_ctrs${contrastive_strategy}_ctra${contrastive_align}_ctrw${contrastive_weight}_ctrt${contrastive_temperature}_patience${patience}


output_dir=$OUTPUT_DIR/${name}/
LOGS_DIR=$OUTPUT_DIR/${name}/





fairseq-train $local_data_dir \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
  --arch $arch  --max-source-positions 256 --max-target-positions 256 \
  --dropout $dropout \
  --weight-decay 0.1  \
  --clip-norm ${clip_norm}   \
  --criterion $criterion --label-smoothing 0.1   \
  --contrastive-strategy ${contrastive_strategy}  --contrastive-weight ${contrastive_weight}  --contrastive-temperature ${contrastive_temperature}  \
  --task raw_video_translation_from_np \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq  \
  --skip-invalid-size-inputs-valid-test \
  --seed $seed \
  --no-progress-bar  \
  --eval-bleu \
  --eval-bleu-args '{"beam": 4,"lenpen":1.0}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --max-epoch ${max_epoches} --keep-interval-updates 10 --keep-best-checkpoints 10  \
  --patience $patience \
  --video-feat-path $video_feat_path \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim \
  --video-feat-type $video_feat_type \
  --max-vid-len $max_vid_len  --train-sampling-strategy ${train_sampling_strategy}   \
  --video-dropout $video_dropout  \
  --fp16  2>&1 | tee -a $LOGS_DIR/train.log

