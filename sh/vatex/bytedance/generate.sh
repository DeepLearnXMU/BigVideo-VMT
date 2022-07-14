

#!/bin/bash

checkpoint_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/vatex/textonly_char_archtransformer_vatex_tgtzh_lr0.001_wu2000_me100_seed1207_gpu1_mt4096_acc1_wd0.1_cn0.0_patience10
local_output_dir=~/fairseq_results


script_root=/opt/tiger/fairseq_mmt/perl
multi_bleu=$script_root/multi-bleu.perl
who=test
test_DATA=~/data/en_zh.char
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
--output $local_output_dir/$checkpoint.$length_penalty.gen-$who.txt

python3 rerank.py $local_output_dir/$checkpoint.$length_penalty.gen-$who.txt $local_output_dir/$checkpoint.$length_penalty.gen-$who.txt.sorted

echo "-----formating json-----"
ids_dir=~/data/raw_texts/ids/test.ids
hypos_dir=$local_output_dir/$checkpoint.$length_penalty.gen-$who.txt.sorted
result_path=$local_output_dir

python3 sh/vatex/construct_json.py $ids_dir $hypos_dir $result_path $checkpoint
echo "-----done-----"


if [ -n "$ensemble" ]; then
      if [ ! -e "$checkpoint_dir/last$ensemble.ensemble.pt" ]; then
              PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $checkpoint_dir --output $checkpoint_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
      fi
      checkpoint=last$ensemble.ensemble.pt
fi

echo "-----done-----"