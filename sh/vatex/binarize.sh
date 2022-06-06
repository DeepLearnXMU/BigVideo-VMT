src='en'
tgt='zh'
TEXT='/home/sata/kly/videoNMT/data/preprocess_follow'
fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train.char \
  --validpref $TEXT/dev.char \
  --testpref $TEXT/test.char \
  --destdir $TEXT/data-bin/${src}_${tgt}.char \
  --workers 8