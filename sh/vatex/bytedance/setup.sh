
#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128


mkdir ~/data
mkdir -p ~/data/vatex/
mkdir -p ~/data/vatex/video/images_resized_r3/vit_base_patch16_224/
mkdir -p ~/data/xigua/fairseq_bin/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/en_zh.char.tar.gz ~/data/en_zh.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/vatex_features.tar.gz ~/data/vatex_features.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/raw_texts.tar.gz ~/data/vatex/raw_texts.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/common.tar.gz /opt/tiger/common.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/video/images_resized_r3/cls.tar.gz ~/data/vatex/video/images_resized_r3/vit_base_patch16_224
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/video/images_resized_r3/patch.tar.gz ~/data/vatex/video/images_resized_r3/vit_base_patch16_224
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/video/slowfast.tar.gz ~/data/vatex/video/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/video/slowfast13.tar.gz ~/data/vatex/video/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/video/clip.tar.gz ~/data/vatex/video/
#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/how2/fairseq_bin/how2_en_pt ~/data/how2/fairseq_bin/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/fairseq_bin_filter.tar.gz ~/data/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/xigua/fairseq_bin/text ~/data/xigua/fairseq_bin/
#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/masking ~/data/vatex/

cd ~/data
tar -zxvf en_zh.tar.gz
tar -zxvf vatex_features.tar.gz
tar -zxvf fairseq_bin_filter.tar.gz
cd ~/data/vatex/
tar -zxvf raw_texts.tar.gz
mv ~/data/vatex/raw_texts/filter_ids/dev.ids ~/data/vatex/raw_texts/filter_ids/valid.ids

cd ~/data/vatex/video/images_resized_r3/vit_base_patch16_224
tar -zxvf cls.tar.gz
tar -zxvf patch.tar.gz
mv cls/dev cls/valid
mv patch/dev patch/valid
cd ~/data/vatex/video/
tar -zxvf slowfast.tar.gz
tar -zxvf slowfast13.tar.gz
tar -zxvf clip.tar.gz
cd /opt/tiger
tar -zxvf common.tar.gz

sudo cp -r /usr/local/bin/pip /usr/bin/pip
sudo cp -r /usr/local/bin/pip3 /usr/bin/pip3
sudo cp -r /usr/local/bin/pip3.7 /usr/bin/pip3.7
pip config set global.index-url https://bytedpypi.byted.org/simple/
cd /opt/tiger/common
pip install --editable ./
cd /opt/tiger/fairseq_mmt
sudo pip install --editable ./
pip install sacremoses
pip install sacrebleu==1.5.1
pip install timm==0.4.12
pip install vizseq==0.1.15
pip install nltk==3.6.4
pip install sacrebleu==1.5.1

pip install -r requirment.txt