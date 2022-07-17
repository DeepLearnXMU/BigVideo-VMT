
#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128


mkdir ~/data
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/en_zh.char.tar.gz ~/data/en_zh.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/vatex_features.tar.gz ~/data/vatex_features.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/raw_texts.tar.gz ~/data/raw_texts.tar.gz
cd ~/data
tar -zxvf en_zh.tar.gz
tar -zxvf vatex_features.tar.gz
tar -zxvf raw_texts.tar.gz

pip config set global.index-url https://bytedpypi.byted.org/simple/
cd /opt/tiger/common
pip install --editable ./
cd /opt/tiger/fairseq_mmt
sudo pip install --editable ./
pip install sacremoses
pip install sacrebleu==1.5.1
pip install timm==0.4.12
pip install vizseq==0.1.15
pip install nltk==3.6.4
pip install sacrebleu==1.5.1

src_lang=en
tgt_lang=zh

local_data_dir=~/data/en_zh.char

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
seed=1207
weight_decay=0.1
clip_norm=0.0
arch=gated_vatex

video_feat_path=~/data/vatex_features
video_ids_path=~/data/raw_texts/ids
video_feat_dim=1024


gpu_num=1


name=vatex_char_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_me${max_epoches}_seed${seed}_gpu${gpu_num}_mt${max_tokens}_acc${update_freq}_wd${weight_decay}_cn${clip_norm}_patience${patience}

output_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex/gated/${name}
LOGS_DIR=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_logs/vatex/gated
local_logs_dir=~/fairseq_logs/vatex/gated

hdfs dfs -mkdir -p $LOGS_DIR
hdfs dfs -mkdir -p $output_dir
mkdir -p $local_logs_dir

hdfs dfs -put ${BASH_SOURCE[0]} $output_dir/train.sh



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
  --fp16  2>&1 | tee -a $local_logs_dir/log.${name}

echo "---put log to $LOGS_DIR/log.${name}---"
hdfs dfs -put $local_logs_dir/log.${name} $LOGS_DIR/log.${name}

