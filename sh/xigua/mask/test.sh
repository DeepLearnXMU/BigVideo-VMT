#!/bin/bash

base_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua
local_output_dir=~/fairseq_results/xigua/
mkdir -p $local_output_dir

mask=mask0
model=video_double_cross_att_base

script_root=~/fairseq_mmt/perl
detokenizer=$script_root/detokenizer.perl
replace_unicode_punctuation=$script_root/replace-unicode-punctuation.perl
tokenizer=$script_root/tokenizer.perl
multi_bleu=$script_root/multi-bleu.perl



mask=mask0
test_DATA=/mnt/bn/luyang/kangliyan/data/xigua/fairseq_bin/text.${mask}



for seed in "27"  "42"

do

  checkpoint_dir=$base_dir/${mask}/${mask}_archvideo_double_cross_att_base_pewln_criCMC_tgtzh_lr7e-4_wu4000_me100_seed${seed}_gpu4_wd0.1_vtypeVIT_cls_rpNone_ia0.0


  who=valid

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
  --task raw_video_translation_from_np   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

  grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
  grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt


  echo "----sacre bleu dtk char level------" >> $local_output_dir/result_log
  perl $detokenizer  -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk
  perl $detokenizer -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
  sacrebleu --tokenize zh $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2 >> $local_output_dir/result_log

  echo "--metor---" >> $local_output_dir/result_log
  echo "----transform zh to char----"
  python3 scripts/transform_zh2char.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
  python3 meteor.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk.char >> $local_output_dir/result_log


  echo "move to $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log

  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log
  fi

  echo "move to$checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log"

  hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
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
  --task raw_video_translation_from_np   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

  grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
  grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt


  echo "----sacre bleu dtk char level------" >> $local_output_dir/result_log
  perl $detokenizer  -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk
  perl $detokenizer -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
  sacrebleu --tokenize zh $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2 >> $local_output_dir/result_log

  echo "--metor---" >> $local_output_dir/result_log
  echo "----transform zh to char----"
  python3 scripts/transform_zh2char.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
  python3 meteor.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk.char >> $local_output_dir/result_log



  echo "move to $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log "
  hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log

  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log
  fi

  echo "move to$checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log"

  hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
  put_result=$?
  if [ $put_result == 1  ]; then
          hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
  fi

done
