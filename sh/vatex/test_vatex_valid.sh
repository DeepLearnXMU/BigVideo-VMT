

export CUDA_VISIBLE_DEVICES=4


checkpoint_dir=/home/sata/kly/fairseq_mmt/output/vatex_gated/vatex_char_archgated_vatex_notop_tgtzh_lr0.001_wu2000_me100_seed1_gpu1_mt4096_acc1_wd0.1_cn0.0_patience10/

script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl
who=valid
test_DATA=/home/sata/kly/videoNMT/data/preprocess_follow/data-bin/en_zh.char
#test_DATA=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh
ensemble=10
video_feat_path=/home/sata/kly/videoNMT/data/vatex_features
video_ids_path=/home/sata/kly/videoNMT/data/raw_texts/ids
video_feat_dim=1024


checkpoint=checkpoint_best.pt
length_penalty=0.8

#echo "-----$who-------"
#fairseq-generate  $test_DATA  \
#--path $checkpoint_dir/$checkpoint \
#--remove-bpe \
#--gen-subset $who \
#--beam 5  \
#--batch-size 128  \
#--lenpen $length_penalty   \
#--output $checkpoint_dir/$checkpoint.$length_length_penalty.gen-$who.txt  | tee $checkpoint_dir/$checkpoint.$length_length_penalty.gen-$who.log



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
--output $checkpoint_dir/$checkpoint.$length_length_penalty.gen-$who.txt  | tee $checkpoint_dir/$checkpoint.$length_length_penalty.gen-$who.log



