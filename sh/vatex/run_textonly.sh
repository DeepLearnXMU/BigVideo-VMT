#! /usr/bin/bash
set -e


device=4
export CUDA_VISIBLE_DEVICES=$device

src_lang=en
tgt_lang=zh

data=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh
criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_epoches=50
dropout=0.3
seed=1
arch=transformer_vatex

name=baseline_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_mt${max_tokens}_patience${patience}

output_dir=/home/sata/kly/fairseq_mmt/output/vatex_baseline/${name}


mkdir -p $output_dir


cp ${BASH_SOURCE[0]} $output_dir/train.sh


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

fairseq-train $data \
  --save-dir $output_dir \
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
  --arch $arch \
  --dropout $dropout \
  --criterion $criterion --label-smoothing 0.1 \
  --task translation \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq --max-epoch $max_epoches \
  --find-unused-parameters \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --patience $patience \
  --keep-last-epochs $keep_last_epochs  \
  --fp16  2>&1 | tee -a $output_dir/train.log

