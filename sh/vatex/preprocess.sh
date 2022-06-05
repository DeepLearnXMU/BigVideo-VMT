
data_dir="/home/sata/kly/videoNMT/data/raw_texts/"

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
  perl $normalize_punctuation -l zh <$DATA/train.zh> $DATA/train.norm.zh
  perl $normalize_punctuation -l en <$DATA/dev.en> $DATA/dev.norm.en
  perl $normalize_punctuation -l zh <$DATA/dev.zh> $DATA/dev.norm.zh
  perl $normalize_punctuation -l en <$DATA/test.en> $DATA/test.norm.en
  perl $normalize_punctuation -l zh <$DATA/test.zh> $DATA/test.norm.zh

  #tokenization
  echo "-----tokenization-------"
  perl $tokenizer -l en <$DATA/train.norm.en> $DATA/train.norm.tok.en
  perl $tokenizer -l zh <$DATA/train.norm.zh> $DATA/train.norm.tok.zh
  perl $tokenizer -l en <$DATA/dev.norm.en> $DATA/dev.norm.tok.en
  perl $tokenizer -l zh <$DATA/dev.norm.zh> $DATA/dev.norm.tok.zh
  perl $tokenizer -l en <$DATA/test.norm.en> $DATA/test.norm.tok.en
  perl $tokenizer -l zh <$DATA/test.norm.zh> $DATA/test.norm.tok.zh

  # remove preprint char
  echo "---- remove preprint char------"
  perl $nonprinting_character_removal -l en <$DATA/train.norm.tok.en> $DATA/train.norm.tok.re.en
  perl $nonprinting_character_removal -l zh <$DATA/train.norm.tok.zh> $DATA/train.norm.tok.re.zh
  perl $nonprinting_character_removal -l en <$DATA/dev.norm.tok.en> $DATA/dev.norm.tok.re.en
  perl $nonprinting_character_removal -l zh <$DATA/dev.norm.tok.zh> $DATA/dev.norm.tok.re.zh
  perl $nonprinting_character_removal -l en <$DATA/test.norm.tok.en> $DATA/test.norm.tok.re.en
  perl $nonprinting_character_removal -l zh <$DATA/test.norm.tok.zh> $DATA/test.norm.tok.re.zh



  # truecase trainmodel
  echo "---- truecase trainmodel------"
  perl $train_truecasing -corpus $DATA/train.norm.tok.re.clean.en -model $DATA/truecase-model.en

  #apply truecase
  echo "---- apply truecase------"
  perl $true_casing -model $DATA/truecase-model.en < $DATA/train.norm.tok.re.clean.en > $DATA/train.norm.tok.re.clean.tc.en
  perl $true_casing -model $DATA/truecase-model.en < $DATA/dev.norm.tok.re.en > $DATA/dev.norm.tok.re.tc.en
  perl $true_casing -model $DATA/truecase-model.en < $DATA/test.norm.tok.re.en > $DATA/test.norm.tok.re.tc.en

