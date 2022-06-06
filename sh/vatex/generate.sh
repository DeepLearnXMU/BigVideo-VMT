

export CUDA_VISIBLE_DEVICES=5

last=5
upper_bound=100000
checkpoint_dir=/home/sata/kly/fairseq_mmt/output/vatex_baseline/baseline_archtransformer_vatex_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.0_patience10


#checkpoint_dir=/home/sata/kly/fairseq/fairseq_output/wmt2016ende/big_teacher

checkpoint=checkpoint_best.pt
#checkpoint=upper300000_last5.pt
who=valid
test_DATA=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh

script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl


echo "-----$who-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size 128  \
--lenpen 0.8   \
--output $checkpoint_dir/gen-$who.txt

python3 rerank.py $checkpoint_dir/gen-$who.txt $checkpoint_dir/gen-$who.txt.sorted

