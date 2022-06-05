# preprocess masking data
src='en'
tgt='de'
mask=mask3
TEXT=data/multi30k-en-$tgt.$mask

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir data-bin/multi30k.en-$tgt.$mask \
  --workers 8 --joined-dictionary \
  --srcdict data/dict.en2de_mask.txt
