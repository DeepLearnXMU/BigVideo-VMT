
DATA="/home/sata/kly/videoNMT/data/raw_texts/"

script_root=/home/kly/fairseq/perl
tokenizer=$script_root/tokenizer.perl

#tokenization
echo "-----tokenization-------"
perl $tokenizer -l en <$DATA/train.norm.en> $DATA/train.norm.tok.en



