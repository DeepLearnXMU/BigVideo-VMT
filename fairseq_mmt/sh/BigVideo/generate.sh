#!/bin/bash






script_root=$YOUR_DIR/fairseq_mmt/perl
detokenizer=$script_root/detokenizer.perl
replace_unicode_punctuation=$script_root/replace-unicode-punctuation.perl
tokenizer=$script_root/tokenizer.perl
multi_bleu=$script_root/multi-bleu.perl



test_DATA=$TEST_BIN


local_output_dir=$OUTPUT_DIR
mkdir -p $local_output_dir

video_ids_path=$DATA_DIR/test
video_feat_type="VIT"
if  [ $video_feat_type == "VIT" ]; then
        video_feat_dim=768
        video_feat_path=$DATA_DIR/video_features/VIT

  elif [ $video_feat_type == "slowfast" ]; then
        video_feat_dim=2304
        video_feat_path=$DATA_DIR/video_features/slowfast
fi

max_vid_len=12

checkpoint_dir=$MODEL_DIR

who=test

checkpoint=checkpoint_best.pt
length_penalty=1.0

echo "-----$who-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 4  \
--batch-size  128  \
--lenpen $length_penalty  \
--video-feat-path $video_feat_path \
--video-ids-path $video_ids_path \
--video-feat-dim $video_feat_dim \
--video-feat-type $video_feat_type \
--max-vid-len $max_vid_len   \
--task raw_video_translation_from_np   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

grep ^S $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.src
grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt



perl $detokenizer  -l en < $local_output_dir/text.$checkpoint.$length_penalty.$who.src > $local_output_dir/text.$checkpoint.$length_penalty.$who.src.dtk

python3 /root/fairseq_mmt/scripts/detokenize_zh.py --input $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo --output $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk
python3 /root/fairseq_mmt/scripts/detokenize_zh.py --input $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt --output $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk

python3 evaluate.py --base_dir $local_output_dir  >> $local_output_dir/result_log


bash /PATHTO/terminology_evaluation/test.sh $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo   >> $local_output_dir/result_log

