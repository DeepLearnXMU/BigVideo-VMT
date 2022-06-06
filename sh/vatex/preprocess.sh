
DATA="/home/sata/kly/videoNMT/data/preprocess_follow"

script_root=/home/kly/fairseq/perl
tokenizer=$script_root/tokenizer.perl

#
tokenization
echo "-----tokenization-------"
perl $tokenizer -l en <$DATA/train.low.en> $DATA/train.tok.en
perl $tokenizer -l en <$DATA/dev.low.en> $DATA/dev.tok.en
perl $tokenizer -l en <$DATA/test.low.en> $DATA/test.tok.en

  ##learn BPE
BPE_ROOT=/home/kly/subword-nmt
python $BPE_ROOT/learn_joint_bpe_and_vocab.py    \
       --input $DATA/train.tok.en  -s 20000 -o $DATA/bpe.20000.en \
       --write-vocabulary $DATA/vocab.20000.en
#apply bpe
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.en  < $DATA/train.tok.en > $DATA/train.char.en
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.en  < $DATA/dev.tok.en > $DATA/dev.char.en
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.en  < $DATA/test.tok.en > $DATA/test.char.en

#python $BPE_ROOT/learn_joint_bpe_and_vocab.py    \
#       --input $DATA/train.tok.zh  -s 10000 -o $DATA/bpe.10000.zh \
#       --write-vocabulary $DATA/vocab.10000.zh
##apply bpe
#python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.zh < $DATA/train.tok.zh > $DATA/train.tok.bpe.zh
#python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.zh < $DATA/dev.tok.zh > $DATA/dev.tok.bpe.zh
#python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000.zh < $DATA/test.tok.zh > $DATA/test.tok.bpe.zh


