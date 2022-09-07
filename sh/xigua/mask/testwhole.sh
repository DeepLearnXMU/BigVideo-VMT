#!/bin/bash

base_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua/
local_output_dir=~/fairseq_results/xigua/
mkdir -p $local_output_dir

mask=mask_verb_1000000
model=video_fushion_encoder_small_merge_before_pewln

script_root=/opt/tiger/fairseq_mmt/perl
detokenizer=$script_root/detokenizer.perl
replace_unicode_punctuation=$script_root/replace-unicode-punctuation.perl
tokenizer=$script_root/tokenizer.perl
multi_bleu=$script_root/multi-bleu.perl



test_DATA=/mnt/bn/luyang/kangliyan/data/xigua/fairseq_bin/xigua.en-zh.${mask}.withtest

video_ids_path=/mnt/bn/luyang/kangliyan/data/xigua/text/preprocessd_v1
video_feat_type="VIT_cls"
if [ $video_feat_type == "VIT_cls"  ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bn/luyang/kangliyan/data/xigua/VIT_cls/
        max_vid_len=32
  elif [ $video_feat_type == "VIT_patch_avg" ]; then
        video_feat_dim=768
        video_feat_path=/mnt/bn/luyang/kangliyan/data/xigua/VIT_patch/
        max_vid_len=197
fi



for mask in "mask0"  "mask1000000" "mask_verb_1000000"

do
  test_DATA=/mnt/bn/luyang/kangliyan/data/xigua/fairseq_bin/xigua.en-zh.${mask}.withtest

  checkpoint_dir=$base_dir/${mask}/${mask}_arch${model}_criCMC_tgtzh_lr7e-4_wu4000_me100_seed1207_gpu4_wd0.1_vtypeVIT_cls_rpNone_ia0.0_patience10


  who=valid

  checkpoint=checkpoint_best.pt
  length_penalty=0.8

  echo "----$mask 1207 $model $who--------" >> $local_output_dir/result_log

  echo "-----$who-------"
  fairseq-generate  $test_DATA  \
  --path $checkpoint_dir/$checkpoint \
  --remove-bpe \
  --gen-subset $who \
  --beam 5  \
  --batch-size  128  \
  --lenpen $length_penalty  \
  --video-feat-path $video_feat_path \
    --video-ids-path $video_ids_path \
    --video-feat-dim $video_feat_dim \
    --video-feat-type $video_feat_type \
    --max-vid-len $max_vid_len   \
    --task raw_video_translation_from_np   | tee $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log

  grep ^H $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo
  grep ^T $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt


  echo "----sacre bleu dtk char level------" >> $local_output_dir/result_log
  perl $detokenizer  -l zh < $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk
  perl $detokenizer -l zh < $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk
  sacrebleu --tokenize zh $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2 >> $local_output_dir/result_log

  echo "--metor---" >> $local_output_dir/result_log
  echo "----transform zh to char----"
  python3 scripts/transform_zh2char.py $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk
  python3 meteor.py $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk.char >> $local_output_dir/result_log


  echo "move to $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log

  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log
  fi

  echo "move to$checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log"

  hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log
  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log
  fi

  who=test

  checkpoint=checkpoint_best.pt
  length_penalty=0.8

  echo "----$mask $seed $model $who--------" >> $local_output_dir/result_log
  echo "-----$who-------"
  fairseq-generate  $test_DATA  \
  --path $checkpoint_dir/$checkpoint \
  --remove-bpe \
  --gen-subset $who \
  --beam 5  \
  --batch-size  128  \
  --lenpen $length_penalty  \
  --video-feat-path $video_feat_path \
    --video-ids-path $video_ids_path \
    --video-feat-dim $video_feat_dim \
    --video-feat-type $video_feat_type \
    --max-vid-len $max_vid_len   \
    --task raw_video_translation_from_np   | tee $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log

  grep ^H $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo
  grep ^T $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt


  echo "----sacre bleu dtk char level------" >> $local_output_dir/result_log
  perl $detokenizer  -l zh < $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk
  perl $detokenizer -l zh < $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk
  sacrebleu --tokenize zh $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2 >> $local_output_dir/result_log

  echo "--metor---" >> $local_output_dir/result_log
  echo "----transform zh to char----"
  python3 scripts/transform_zh2char.py $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk
  python3 meteor.py $local_output_dir/$model.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/$model.$checkpoint.$length_penalty.$who.tgt.dtk.char >> $local_output_dir/result_log



  echo "move to $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log

  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.gen-$who.log
  fi

  echo "move to$checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log"

  hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log
  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/$model.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/$model.$checkpoint.$length_penalty.meteor_$who.log
  fi

done
