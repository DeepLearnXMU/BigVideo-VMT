
export CUDA_VISIBLE_DEVICES=4,5
python3 scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path  /home/sata/kly/fairseq_mmt-main/data/flickr30k --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset val --model vit_base_patch16_384 --path  /home/sata/kly/fairseq_mmt-main/data/flickr30k --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset test2016 --model vit_base_patch16_384 --path  /home/sata/kly/fairseq_mmt-main/data/flickr30k --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset test2017 --model vit_base_patch16_384 --path  /home/sata/kly/fairseq_mmt-main/data/flickr30k --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset testcoco --model vit_base_patch16_384 --path  /home/sata/kly/fairseq_mmt-main/data/flickr30k --save /home/sata/kly/fairseq_mmt/img_feature/




