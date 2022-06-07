

export CUDA_VISIBLE_DEVICES=5


checkpoint_dir=/home/sata/kly/fairseq_mmt/output/vatex_baseline/textonly_char_archtransformer_tiny_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.1_patience10

script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl
who=valid
test_DATA=/home/sata/kly/videoNMT/data/preprocess_follow/data-bin/en_zh.char
ensemble=10

checkpoint=checkpoint_best.pt

echo "-----$who-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size 128  \
--lenpen 0.8   \
--output $checkpoint_dir/$checkpoint.gen-$who.txt   > $checkpoint_dir/$checkpoint.gen-$who.log


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
--batch-size 128  \
--lenpen 0.8   \
--output $checkpoint_dir/$checkpoint.gen-$who.txt  > $checkpoint_dir/$checkpoint.gen-$who.log



