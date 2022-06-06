

export CUDA_VISIBLE_DEVICES=5


checkpoint_dir=/home/sata/kly/fairseq_mmt/output/vatex_baseline/baseline_archtransformer_vatex_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.1_patience10

script_root=/home/kly/fairseq/perl
multi_bleu=$script_root/multi-bleu.perl
who=test
test_DATA=/home/sata/kly/videoNMT/data/raw_texts/data-bin/en_zh
ensemble=10

checkpoint=checkpoint_best.pt


if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi



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

echo "-----formating json-----"
ids_dir="/home/sata/kly/videoNMT/data/raw_texts/test.ids"
hypos_dir=$checkpoint_dir
result_path=$checkpoint_dir

python3 /home/kly/fairseq_mmt/sh/vatex/construct_json.py $ids_dir $hypos_dir $result_path
echo "-----done-----"