#! /usr/bin/bash
set -e


device=4
export CUDA_VISIBLE_DEVICES=$device
source activate fairseq_mmt

src_lang=en
tgt_lang=zh


#data=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh
data=/home/sata/kly/videoNMT/data/preprocess_follow/data-bin/en_zh.char
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
seed=1
weight_decay=0.1
clip_norm=0.0
arch=vatex_multimodal_transformer_att_vatex_top_pe_prenorm

video_feat_path=/home/sata/kly/videoNMT/data/vatex_features
video_ids_path=/home/sata/kly/videoNMT/data/raw_texts/ids
video_feat_dim=1024
SA_attention_dropout=0.1
SA_video_dropout=0.1


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`


name=vatex_char_arch${arch}detach_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}_cn${clip_norm}_patience${patience}_avdp${SA_video_dropout}_aadp${SA_attention_dropout}

output_dir=/home/sata/kly/fairseq_mmt/output/${arch}/${name}


mkdir -p $output_dir


cp ${BASH_SOURCE[0]} $output_dir/train.sh




fairseq-train $data \
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
  --SA-video-dropout ${SA_video_dropout} --SA-attention-dropout ${SA_attention_dropout} \
  --fp16  2>&1 | tee -a $output_dir/train.log


