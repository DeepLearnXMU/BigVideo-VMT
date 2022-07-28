
#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128


#mkdir ~/data
#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/en_zh.char.tar.gz ~/data/en_zh.tar.gz
#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/vatex_features.tar.gz ~/data/vatex_features.tar.gz
#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/raw_texts.tar.gz ~/data/raw_texts.tar.gz
#cd ~/data
#tar -zxvf en_zh.tar.gz
#tar -zxvf vatex_features.tar.gz
#tar -zxvf raw_texts.tar.gz
#
#pip config set global.index-url https://bytedpypi.byted.org/simple/
#cd /opt/tiger/common
#pip install --editable ./
#cd /opt/tiger/fairseq_mmt
#sudo pip install --editable ./
#pip install sacremoses
#pip install sacrebleu==1.5.1
#pip install timm==0.4.12
#pip install vizseq==0.1.15
#pip install nltk==3.6.4
#pip install sacrebleu==1.5.1

export CUDA_VISIBLE_DEVICES=0


base_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex_0727/fushion
local_output_dir=~/fairseq_results
mkdir -p $local_output_dir
script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl
who=valid
test_DATA=~/data/en_zh.char
ensemble=10
video_feat_path=~/data/vatex_features
video_ids_path=~/data/raw_texts/ids
video_feat_dim=1024
length_penalty=0.8

for item in "vatex_char_archvatex_fushion_small_after_pewln_criCMC_tgtzh_lr0.001_wu2000_me100_seed1207_gpu1_mt4096_acc1_wd0.1_cn0.0_patience10" \
            "vatex_char_archvatex_fushion_small_after_pewoln_criCMC_tgtzh_lr0.001_wu2000_me100_seed1207_gpu1_mt4096_acc1_wd0.1_cn0.0_patience10" \


do

  echo "----validating best $item--------"
  checkpoint_dir=$base_dir/${item}
  checkpoint=checkpoint_best.pt


  echo "-----$who-------"
  fairseq-generate  $test_DATA  \
  --path $checkpoint_dir/$checkpoint \
  --remove-bpe \
  --gen-subset $who \
  --beam 5  \
  --batch-size  128  \
  --lenpen $length_penalty  \
  --task vatex_translation \
  --video-feat-path $video_feat_path \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim \
  --output $local_output_dir/$checkpoint.$length_penalty.gen-$who.log | tee $local_output_dir/$checkpoint.$length_penalty.gen-$who.log

  echo "move to $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put $local_output_dir/$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log


  echo "----validating ensemble10  $item--------"

  if [ -n "$ensemble" ]; then
      if [ ! -e "$checkpoint_dir/last$ensemble.ensemble.pt" ]; then
              PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $checkpoint_dir --output $checkpoint_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
      fi
      checkpoint=last$ensemble.ensemble.pt
  fi


  echo "-----$who ensemble-------"
  fairseq-generate  $test_DATA  \
  --path $checkpoint_dir/$checkpoint \
  --remove-bpe \
  --gen-subset $who \
  --beam 5  \
  --batch-size  128  \
  --lenpen $length_penalty  \
  --task vatex_translation \
  --video-feat-path $video_feat_path \
  --video-ids-path $video_ids_path \
  --video-feat-dim $video_feat_dim \
  --output $local_output_dir/$checkpoint.$length_penalty.gen-$who.log | tee $local_output_dir/$checkpoint.$length_penalty.gen-$who.log

  echo "move to $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put $local_output_dir/$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$checkpoint.$length_penalty.gen-$who.log



done


