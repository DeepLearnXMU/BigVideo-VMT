

for domain in "medical" "law" "it" "koran" "subtitles"
do
   echo "-----domain is $domain-------"
  DATA=/home/sata/kly/data/multidomain/$domain


  script_root=/home/kly/fairseq/perl
  normalize_punctuation=$script_root/normalize-punctuation.perl
  tokenizer=$script_root/tokenizer.perl
  nonprinting_character_removal=$script_root/remove-non-printing-char.perl
  train_truecasing=$script_root/train-truecaser.perl
  true_casing=$script_root/truecase.perl
  clean=$script_root/clean-corpus-n.perl

  #normalize-punctuation
  echo "-----normalize-punctuation-------"
  perl $normalize_punctuation -l en <$DATA/train.en> $DATA/train.norm.en
  perl $normalize_punctuation -l de <$DATA/train.de> $DATA/train.norm.de
  perl $normalize_punctuation -l en <$DATA/dev.en> $DATA/dev.norm.en
  perl $normalize_punctuation -l de <$DATA/dev.de> $DATA/dev.norm.de
  perl $normalize_punctuation -l en <$DATA/test.en> $DATA/test.norm.en
  perl $normalize_punctuation -l de <$DATA/test.de> $DATA/test.norm.de

  #tokenization
  echo "-----tokenization-------"
  perl $tokenizer -l en <$DATA/train.norm.en> $DATA/train.norm.tok.en
  perl $tokenizer -l de <$DATA/train.norm.de> $DATA/train.norm.tok.de
  perl $tokenizer -l en <$DATA/dev.norm.en> $DATA/dev.norm.tok.en
  perl $tokenizer -l de <$DATA/dev.norm.de> $DATA/dev.norm.tok.de
  perl $tokenizer -l en <$DATA/test.norm.en> $DATA/test.norm.tok.en
  perl $tokenizer -l de <$DATA/test.norm.de> $DATA/test.norm.tok.de

  # remove preprint char
  echo "---- remove preprint char------"
  perl $nonprinting_character_removal -l en <$DATA/train.norm.tok.en> $DATA/train.norm.tok.re.en
  perl $nonprinting_character_removal -l de <$DATA/train.norm.tok.de> $DATA/train.norm.tok.re.de
  perl $nonprinting_character_removal -l en <$DATA/dev.norm.tok.en> $DATA/dev.norm.tok.re.en
  perl $nonprinting_character_removal -l de <$DATA/dev.norm.tok.de> $DATA/dev.norm.tok.re.de
  perl $nonprinting_character_removal -l en <$DATA/test.norm.tok.en> $DATA/test.norm.tok.re.en
  perl $nonprinting_character_removal -l de <$DATA/test.norm.tok.de> $DATA/test.norm.tok.re.de

  #clean
  echo "---- clean------"
  perl $clean $DATA/train.norm.tok.re en de $DATA/train.norm.tok.re.clean 1 100

  # truecase trainmodel
  echo "---- truecase trainmodel------"
  perl $train_truecasing -corpus $DATA/train.norm.tok.re.clean.en -model $DATA/truecase-model.en
  perl $train_truecasing -corpus $DATA/train.norm.tok.re.clean.de -model $DATA/truecase-model.de
  #apply truecase
  echo "---- apply truecase------"
  perl $true_casing -model $DATA/truecase-model.en < $DATA/train.norm.tok.re.clean.en > $DATA/train.norm.tok.re.clean.tc.en
  perl $true_casing -model $DATA/truecase-model.en < $DATA/dev.norm.tok.re.en > $DATA/dev.norm.tok.re.tc.en
  perl $true_casing -model $DATA/truecase-model.en < $DATA/test.norm.tok.re.en > $DATA/test.norm.tok.re.tc.en

  perl $true_casing -model $DATA/truecase-model.de < $DATA/train.norm.tok.re.clean.de > $DATA/train.norm.tok.re.clean.tc.de
  perl $true_casing -model $DATA/truecase-model.de < $DATA/dev.norm.tok.re.de > $DATA/dev.norm.tok.re.tc.de
  perl $true_casing -model $DATA/truecase-model.de < $DATA/test.norm.tok.re.de > $DATA/test.norm.tok.re.tc.de

  cp -r $DATA/train.norm.tok.re.clean.tc.en  /home/sata/kly/data/multidomain/whole_all6/$domain.train.en
  cp -r $DATA/dev.norm.tok.re.tc.en /home/sata/kly/data/multidomain/whole_all6/$domain.dev.en
  cp -r $DATA/test.norm.tok.re.tc.en /home/sata/kly/data/multidomain/whole_all6/$domain.test.en

  cp -r $DATA/train.norm.tok.re.clean.tc.de /home/sata/kly/data/multidomain/whole_all6/$domain.train.de
  cp -r $DATA/dev.norm.tok.re.tc.de  /home/sata/kly/data/multidomain/whole_all6/$domain.dev.de
  cp -r $DATA/test.norm.tok.re.tc.de /home/sata/kly/data/multidomain/whole_all6/$domain.test.de




#  ##learn BPE
#  BPE_ROOT=/home/kly/subword-nmt
#  python $BPE_ROOT/learn_joint_bpe_and_vocab.py    \
#         --input $DATA/train.norm.tok.re.clean.tc.en $DATA/train.norm.tok.re.clean.tc.de -s 32000 -o $DATA/bpe.32000 \
#         --write-vocabulary $DATA/vocab.32000.en $DATA/vocab.32000.de
#  #apply bpe
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/train.norm.tok.re.clean.tc.en > $DATA/train.norm.tok.re.clean.tc.bpe.en
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/train.norm.tok.re.clean.tc.de > $DATA/train.norm.tok.re.clean.tc.bpe.de
#
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/dev.norm.tok.re.tc.en > $DATA/dev.norm.tok.re.tc.bpe.en
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/dev.norm.tok.re.tc.de > $DATA/dev.norm.tok.re.tc.bpe.de
#
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/test.norm.tok.re.tc.en > $DATA/test.norm.tok.re.tc.bpe.en
#  python $BPE_ROOT/apply_bpe.py -c $DATA/bpe.32000 < $DATA/test.norm.tok.re.tc.de > $DATA/test.norm.tok.re.tc.bpe.de



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

done
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