
for mask in "mask35565" "mask_verb_35565"   \

do
    src='en'
    tgt='zh'
    echo "====$mask===="
    TEXT=~/data/vatex/masking/$mask

    fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir  ~/data/fairseq_bin/vatex.en-$tgt.$mask \
    --workers 8
done