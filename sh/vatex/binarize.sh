src='en'
tgt='zh'
TEXT='/home/sata/kly/videoNMT/data/raw_texts'
fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train.tok.bpe \
  --validpref $TEXT/valid.tok.bpe \
  --testpref $TEXT/test.tok.bpe \
  --destdir $TEXT/data-bin/src_tgt \
  --workers 8