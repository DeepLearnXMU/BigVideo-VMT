
DATA="/home/sata/kly/videoNMT/data/raw_texts"

script_root=/home/kly/fairseq/perl
tokenizer=$script_root/tokenizer.perl

#
#tokenization
#echo "-----tokenization-------"
#perl $tokenizer -l en <$DATA/train.en> $DATA/train.tok.en
#perl $tokenizer -l en <$DATA/dev.en> $DATA/dev.tok.en
#perl $tokenizer -l en <$DATA/test.en> $DATA/test.tok.en

  ##learn BPE
BPE_ROOT=/home/kly/subword-nmt
python $BPE_ROOT/learn_joint_bpe_and_vocab.py    \
       --input $DATA/train.tok.en  -s 10000 -o $DATA/bpe.10000 \
       --write-vocabulary $DATA/vocab.10000.en
#apply bpe
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000 < $DATA/train.tok.en > $DATA/train.tok.bpe.en
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000 < $DATA/dev.tok.en > $DATA/dev.tok.bpe.en
python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.10000 < $DATA/test.tok.en > $DATA/test.tok.bpe.en

