This code repository is for the accepted ACL2022 paper "On Vision Features in Multimodal Machine Translation". We provide the details and scripts for the proposed probing tasks. We hope the code could help those who want to research on the multimodal machine translation task.
# Our dependency

* PyTorch version == 1.9.1
* Python version == 3.6.7
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1

# Install fairseq

```bash
cd fairseq_mmt
pip install --editable ./
```

# Multi30k data & Flickr30k entities
Multi30k data from [here](https://github.com/multi30k/dataset) and [here](https://www.statmt.org/wmt17/multimodal-task.html)  
flickr30k entities data from [here](https://github.com/BryanPlummer/flickr30k_entities)  
Here, We get multi30k text data from [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)
```bash
cd fairseq_mmt
git clone https://github.com/BryanPlummer/flickr30k_entities.git
cd flickr30k_entities
unzip annotations.zip

# download data and create a directory anywhere
flickr30k
├─ flickr30k-images
├─ test2017-images
├─ test_2016_flickr.txt
├─ test_2017_flickr.txt
├─ test_2017_mscoco.txt
├─ test_2018_flickr.txt
├─ testcoco-images
├─ train.txt
└─ val.txt
```

# Extract image feature
```bash
# please read scripts/README.md firstly
python3 scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path ../flickr30k
```
script parameters:
- ```dataset```: choices=['train', 'val', 'test2016', 'test2017', 'testcoco']
- ```model```:  'vit_base_patch16_384', that you can find in [timm.list_models()](https://github.com/rwightman/pytorch-image-models/)
- ```path```:    '/path/to/your/flickr30k'

# Create masking data
```bash
pip3 install stanfordcorenlp 
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip

cd fairseq_mmt
cat data/multi30k/train.en data/multi30k/valid.en data/multi30k/test.2016.en > train_val_test2016.en
python3 get_and_record_noun_from_f30k_entities.py # recording noun and nouns position in each sentence by flickr30k_entities
python3 record_color_people_position.py

# create en-de masking data
cd data/masking
python3 match_origin2bpe_position.py
python3 create_masking_multi30k.py         # create mask1-4 & color & people data 

sh preprocess_mmt.sh
```

# Train and Test
#### 1. Preprocess(mask1 as an example)
```bash
src='en'
tgt='de'
mask=mask1  # mask1, mask2, mask3, maskc(color), maskp(character)
TEXT=data/multi30k-en-$tgt.$mask

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir data-bin/multi30k.en-$tgt.$mask \
  --workers 8 --joined-dictionary \
  --srcdict data/dict.en2de_$mask.txt
```
*sh preprocess.sh to generate no masking data*
#### 2. Train(mask1 as an example)
```bash
mask_data=mask1
data_dir=multi30k.en-de.mask1
src_lang='en'
tgt_lang='de'
image_feat=vit_base_patch16_384
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/multi30k-en2de/$tag
image_feat_path=data/$image_feat
image_feat_dim=768

criterion=label_smoothed_cross_entropy
fp16=1
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=8000
dropout=0.3

arch=image_multimodal_transformer_SA_top
SA_attention_dropout=0.1
SA_image_dropout=0.1
SA_text_dropout=0

CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size 2 -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --find-unused-parameters
  --share-all-embeddings
  --patience $patience
  --keep-last-epochs $keep_last_epochs
  --SA-image-dropout $SA_image_dropout
  --SA-attention-dropout $SA_attention_dropout
  --SA-text-dropout $SA_text_dropout
```
*you can run train_mmt.sh instead of scripts above.*
#### 3. Test(mask1 as an example)
```bash
#sh translation_mmt.sh $1 $2
sh translation_mmt.sh mask1 vit_base_patch16_384  # origin text is mask0
```
script parameters:
- ```$1```: choices=['mask1', 'mask2', 'mask3', 'mask4', 'maskc', 'maskp', 'mask0']
- ```$2```:  'vit_base_patch16_384', that you can find in [timm.list_models()](https://github.com/rwightman/pytorch-image-models/)

# Visualization
```bash
# uncomment line429-431,487-488 in /fairseq/models/image_multimodal_transformer_SA.py
# decode again, generate tensors to the checkpoint dir
# prepare files needed in /visualization/visualization.py
cd visualization
python3 visualization.py
```
