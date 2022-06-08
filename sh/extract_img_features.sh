
export CUDA_VISIBLE_DEVICES=4,5
python3 scripts/get_img_feat.py --dataset train --model resnet50 --path /home/sata/kly/fairseq_mmt/data/flickr30k/ --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset val --model resnet50 --path /home/sata/kly/fairseq_mmt/data/flickr30k/ --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset test2016 --model resnet50 --path /home/sata/kly/fairseq_mmt/data/flickr30k/ --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset test2017 --model resnet50 --path /home/sata/kly/fairseq_mmt/data/flickr30k/ --save /home/sata/kly/fairseq_mmt/img_feature/

python3 scripts/get_img_feat.py --dataset testcoco --modelresnet50 --path /home/sata/kly/fairseq_mmt/data/flickr30k/ --save /home/sata/kly/fairseq_mmt/img_feature/




