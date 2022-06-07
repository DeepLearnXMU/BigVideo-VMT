src='en'
tgt='fr'

TEXT=/home/sata/kly/fairseq_mmt/data/multi30k-en-$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir /home/sata/kly/fairseq_mmt/data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary


src='en'
tgt='de'

TEXT=/home/sata/kly/fairseq_mmt/data/multi30k-en-$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir /home/sata/kly/fairseq_mmt/data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary
