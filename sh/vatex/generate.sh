

export CUDA_VISIBLE_DEVICES=5

last=5
upper_bound=100000
checkpoint_dir=/home/sata/kly/fairseq_mmt/output/vatex_baseline/baseline_archtransformer_vatex_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.1_patience10


#checkpoint_dir=/home/sata/kly/fairseq/fairseq_output/wmt2016ende/big_teacher

checkpoint=checkpoint_best.pt
#checkpoint=upper300000_last5.pt
who=valid
test_DATA=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh

script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl


echo "-----$who-------"
fairseq-generate  generate.py $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size 128  \
--lenpen 0.8  > $checkpoint_dir/$checkpoint.$who.zh.out

#bash $scripts/compound_split_bleu.sh $checkpoint_dir/$checkpoint.nist14ende.de.out

#grep ^T $checkpoint_dir/$checkpoint.nist14ende.de.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $checkpoint_dir/$checkpoint.nist14ende.de.ref
#grep ^H $checkpoint_dir/$checkpoint.nist14ende.de.out | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $checkpoint_dir/$checkpoint.nist14ende.de.sys
#
#perl $perl/multi-bleu.perl $checkpoint_dir/$checkpoint.nist14ende.de.ref <  $checkpoint_dir/$checkpoint.nist14ende.de.sys
#sacrebleu $checkpoint_dir/$checkpoint.nist13ende.de.ref <  $checkpoint_dir/$checkpoint.nist13ende.de.sys

grep ^H $checkpoint_dir/$checkpoint.$who.zh.out | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $checkpoint_dir/$checkpoint.$who.zh.out.sys
grep ^T $checkpoint_dir/$checkpoint.$who.zh.out | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $checkpoint_dir/$checkpoint.$who.zh.out.ref
SYS=$checkpoint_dir/$checkpoint.nist14ende.de.sys
REF=$checkpoint_dir/$checkpoint.nist14ende.de.ref
