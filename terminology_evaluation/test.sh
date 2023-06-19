python3  evaluate_term.py \
    --language zh \
    --hypothesis $1 \
    --source $DATA_DIR/text_data/test.tok.en  \
    --target_reference $DATA_DIR/text_data/test.tok.zh  \
    --const  $DATA_DIR/text_data/test.anno.combine  \
    --log  test.log

