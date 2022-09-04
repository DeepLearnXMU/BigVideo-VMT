#!/bin/bash

checkpoint_dir=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/fairseq_mmt/fairseq_output/xigua/archtransformer_vaswani_wmt_en_de_big_tgtzh_lr7e-4_wu4000_mu200000_seed1207_gpu4_mt8192_acc1_wd0.0_cn0.0_patience-1
local_output_dir=~/fairseq_results/text
mkdir -p $local_output_dir

script_root=~/fairseq_mmt/perl
detokenizer=$script_root/detokenizer.perl
replace_unicode_punctuation=$script_root/replace-unicode-punctuation.perl
tokenizer=$script_root/tokenizer.perl
multi_bleu=$script_root/multi-bleu.perl



mask=mask0
if [ $mask == "mask0" ]; then
        test_DATA=/root/data/xigua/fairseq_bin/text.withtest
  else
        test_DATA=/mnt/bn/luyang/kangliyan/data/xigua/fairseq_bin/text.${mask}
fi




who=valid

checkpoint=checkpoint_best.pt
length_penalty=0.8

echo "-----$who-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size  128  \
--lenpen $length_penalty  \
--task translation   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt


echo "----sacre bleu dtk char level------"
perl $detokenizer  -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk
perl $detokenizer -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
sacrebleu --tokenize zh $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2

echo "--metor---"
echo "----transform zh to char----"
python3 scripts/transform_zh2char.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
python3 meteor.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk.char | tee $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log



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

echo "-----$who-------"
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 5  \
--batch-size  128  \
--lenpen $length_penalty  \
--task translation   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt


echo "----sacre bleu dtk char level------"
perl $detokenizer  -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk
perl $detokenizer -l zh < $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt >  $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
sacrebleu --tokenize zh $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk -i $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk -m bleu -b -w 2

echo "--metor---"
echo "----transform zh to char----"
python3 scripts/transform_zh2char.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk
python3 meteor.py $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk.char $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk.char | tee $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log



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
# ensemble=10

# if [ -n "$ensemble" ]; then
#       if [ ! -e "$checkpoint_dir/last$ensemble.ensemble.pt" ]; then
#               PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $checkpoint_dir --output $checkpoint_dir/last$ensemble.ensemble.pt --num-update-epoches $ensemble
#       fi
#       checkpoint=last$ensemble.ensemble.pt
# fi


# echo "-----ensemble $who-------"
# fairseq-generate  $test_DATA  \
# --path $checkpoint_dir/$checkpoint \
# --remove-bpe \
# --gen-subset $who \
# --beam 5  \
# --batch-size  128  \
# --lenpen $length_penalty  \
# --task translation   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

# grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.hypo
# grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.tgt


# echo "--sacrebleu---"
# sacrebleu $local_output_dir/text.$checkpoint.$length_penalty.tgt -i $local_output_dir/text.$checkpoint.$length_penalty.hypo -m bleu -b -w 2

# echo "--metor---"
# python3 meteor.py $local_output_dir/text.$checkpoint.$length_penalty.hypo $local_output_dir/text.$checkpoint.$length_penalty.tgt > $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log

# echo "move to $checkpoint_dir/$method.vitcls.$checkpoint.$length_penalty.gen-$who.log "
# hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log

# put_result=$?
# if [ $put_result == 1  ]; then
#         hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.gen-$who.log
# fi

# hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
# put_result=$?
# if [ $put_result == 1  ]; then
#         hdfs dfs -put -f $local_output_dir/text.$checkpoint.$length_penalty.meteor_$who.log $checkpoint_dir/text.$checkpoint.$length_penalty.meteor_$who.log
# fi
