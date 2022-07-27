


DATA=~/data/how2/text_processed
mkdir $DATA/cleaned

#script_root=/opt/tiger/fairseq_mmt/perl
#normalize_punctuation=$script_root/normalize-punctuation.perl
#tokenizer=$script_root/tokenizer.perl
#nonprinting_character_removal=$script_root/remove-non-printing-char.perl
#train_truecasing=$script_root/train-truecaser.perl
#true_casing=$script_root/truecase.perl
#clean=$script_root/clean-corpus-n.perl
#
##normalize-punctuation
#echo "-----normalize-punctuation-------"
#perl $normalize_punctuation -l en <$DATA/train.en> $DATA/train.norm.en
#perl $normalize_punctuation -l de <$DATA/train.pt> $DATA/train.norm.pt
#perl $normalize_punctuation -l en <$DATA/val.en> $DATA/val.norm.en
#perl $normalize_punctuation -l de <$DATA/val.pt> $DATA/val.norm.pt
#perl $normalize_punctuation -l en <$DATA/dev5.en> $DATA/dev5.norm.en
#perl $normalize_punctuation -l de <$DATA/dev5.pt> $DATA/dev5.norm.pt
#
##tokenization
#echo "-----tokenization-------"
#perl $tokenizer -l en <$DATA/train.norm.en> $DATA/train.norm.tok.en
#perl $tokenizer -l pt <$DATA/train.norm.pt> $DATA/train.norm.tok.pt
#perl $tokenizer -l en <$DATA/val.norm.en> $DATA/val.norm.tok.en
#perl $tokenizer -l pt <$DATA/val.norm.pt> $DATA/val.norm.tok.pt
#perl $tokenizer -l en <$DATA/dev5.norm.en> $DATA/dev5.norm.tok.en
#perl $tokenizer -l pt <$DATA/dev5.norm.pt> $DATA/dev5.norm.tok.pt
#
## remove preprint char
#echo "---- remove preprint char------"
#perl $nonprinting_character_removal -l en <$DATA/train.norm.tok.en> $DATA/train.norm.tok.re.en
#perl $nonprinting_character_removal -l pt <$DATA/train.norm.tok.pt> $DATA/train.norm.tok.re.pt
#perl $nonprinting_character_removal -l en <$DATA/val.norm.tok.en> $DATA/val.norm.tok.re.en
#perl $nonprinting_character_removal -l pt <$DATA/val.norm.tok.pt> $DATA/val.norm.tok.re.pt
#perl $nonprinting_character_removal -l en <$DATA/dev5.norm.tok.en> $DATA/dev5.norm.tok.re.en
#perl $nonprinting_character_removal -l pt <$DATA/dev5.norm.tok.pt> $DATA/dev5.norm.tok.re.pt
#
## #clean
## echo "---- clean------"
## perl $clean $DATA/train.norm.tok.re en de $DATA/train.norm.tok.re.clean 1 100
#
## truecase trainmodel
#echo "---- truecase trainmodel------"
#perl $train_truecasing -corpus $DATA/train.norm.tok.re.en -model $DATA/truecase-model.en
#perl $train_truecasing -corpus $DATA/train.norm.tok.re.pt -model $DATA/truecase-model.pt
##apply truecase
#echo "---- apply truecase------"
#perl $true_casing -model $DATA/truecase-model.en < $DATA/train.norm.tok.re.en > $DATA/train.norm.tok.re.tc.en
#perl $true_casing -model $DATA/truecase-model.en < $DATA/val.norm.tok.re.en > $DATA/val.norm.tok.re.tc.en
#perl $true_casing -model $DATA/truecase-model.en < $DATA/dev5.norm.tok.re.en > $DATA/dev5.norm.tok.re.tc.en
#
#perl $true_casing -model $DATA/truecase-model.pt < $DATA/train.norm.tok.re.pt > $DATA/train.norm.tok.re.tc.pt
#perl $true_casing -model $DATA/truecase-model.pt < $DATA/val.norm.tok.re.pt > $DATA/val.norm.tok.re.tc.pt
#perl $true_casing -model $DATA/truecase-model.pt < $DATA/dev5.norm.tok.re.pt > $DATA/dev5.norm.tok.re.tc.pt
#
#cp -r $DATA/train.norm.tok.re.tc.en  $DATA/cleaned/train.en
#cp -r $DATA/val.norm.tok.re.tc.en $DATA/cleaned/val.en
#cp -r $DATA/dev5.norm.tok.re.tc.en $DATA/cleaned/dev5.en
#
#cp -r $DATA/train.norm.tok.re.tc.pt $DATA/cleaned/train.pt
#cp -r $DATA/val.norm.tok.re.tc.pt  $DATA/cleaned/val.pt
#cp -r $DATA/dev5.norm.tok.re.tc.pt $DATA/cleaned/dev5.pt




  ##learn BPE
  BPE_ROOT=/home/kly/subword-nmt
  python $BPE_ROOT/learn_joint_bpe_and_vocab.py    \
         --input $DATA/train.norm.tok.re.clean.tc.en $DATA/train.norm.tok.re.clean.tc.de -s 32000 -o $DATA/bpe.32000 \
         --write-vocabulary $DATA/vocab.32000.en $DATA/vocab.32000.de
  #apply bpe
  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/train.norm.tok.re.clean.tc.en > $DATA/train.norm.tok.re.clean.tc.bpe.en
  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/train.norm.tok.re.clean.tc.de > $DATA/train.norm.tok.re.clean.tc.bpe.de

  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/dev.norm.tok.re.tc.en > $DATA/dev.norm.tok.re.tc.bpe.en
  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/dev.norm.tok.re.tc.de > $DATA/dev.norm.tok.re.tc.bpe.de

  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/test.norm.tok.re.tc.en > $DATA/test.norm.tok.re.tc.bpe.en
  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/test.norm.tok.re.tc.de > $DATA/test.norm.tok.re.tc.bpe.de



#  #binarize
#  mkdir -p $DATA/final
#  ln -srf $DATA/train.norm.tok.re.clean.tc.bpe.en $DATA/final/train.en
#  ln -srf $DATA/train.norm.tok.re.clean.tc.bpe.de $DATA/final/train.de
#  ln -srf $DATA/dev.norm.tok.re.tc.en $DATA/final/valid.en
#  ln -srf $DATA/dev.norm.tok.re.tc.de $DATA/final/valid.de
#  ln -srf $DATA/test.norm.tok.re.tc.en $DATA/final/test.en
#  ln -srf $DATA/test.norm.tok.re.tc.en $DATA/final/test.de
#
#  echo -e "| preprocess ... it will take some time \n"
#  bin_data=$DATA/final
#  bin_dir=/home/sata/kly/data/multidomain/fairseq_bin
#  python fairseq_cli/preprocess.py \
#  --source-lang en --target-lang de \
#  --trainpref $bin_data/train \
#  --validpref $bin_data/valid \
#  --testpref $bin_data/test \
#   --destdir $bin_dir/$domain \
#   --joined-dictionary \
#   --thresholdsrc 2 --thresholdtgt 2


##tokenize the decodes file by moses tokenizer.perl
#perl $tokenizer -l en < $REF.dtk.punc > $REF.dtk.punc.tok
#perl $tokenizer -l de < $REF.dtk.punc > $REF.dtk.punc.tok
#
#python fairseq_cli/preprocess.py --source-lang en --target-lang de \
#  --trainpref $DATA/train \
#  --validpref $DATA/valid \
#  --testpref $DATA/test \
#  --destdir $OUTPUT \
#  --nwordssrc 32768 --nwordstgt 32768 \
#  --joined-dictionary \
#  --workers 20

#for domain in "it" "law" "koran"
#do
#   echo "-----domain is $domain-------"
#  DATA=/home/sata/kly/data/multidomain/$domain
#  cp -r $DATA/train.norm.tok.re.clean.tc.en whole/$domain.train.en
#  cp -r $DATA/train.norm.tok.re.clean.tc.de whole/$domain.train.de
#  cp -r $DATA/dev.norm.tok.re.tc.en whole/$domain.dev.en
#  cp -r $DATA/dev.norm.tok.re.tc.de whole/$domain.dev.de
#  cp -r $DATA/test.norm.tok.re.tc.en whole/$domain.test.en
#  cp -r $DATA/test.norm.tok.re.tc.de whole/$domain.test.de
#done