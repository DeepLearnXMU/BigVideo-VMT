
#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128


#sudo apt-get update
#sudo apt-get install ffmpeg libsm6 libxext6

mkdir ~/data
mkdir -p ~/data/vatex/
mkdir ~/codes

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/common.tar.gz /opt/tiger/common.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/vatex/fairseq_bin_filter.tar.gz ~/data/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/codes/timm.tar.gz ~/codes

cd ~/data
tar -zxvf en_zh.tar.gz
tar -zxvf vatex_features.tar.gz
tar -zxvf fairseq_bin_filter.tar.gz
cd /opt/tiger
tar -zxvf common.tar.gz

sudo cp -r /usr/local/bin/pip /usr/bin/pip
sudo cp -r /usr/local/bin/pip3 /usr/bin/pip3
sudo cp -r /usr/local/bin/pip3.7 /usr/bin/pip3.7
/usr/bin/python3 -m pip install --upgrade pip

pip config set global.index-url https://bytedpypi.byted.org/simple/
cd /opt/tiger/common
pip install --editable ./

cd /opt/tiger/fairseq_mmt

pip install -r requirment.txt

sudo pip install --editable ./
pip install sacremoses
pip install sacrebleu==1.5.1
#pip install timm==0.4.12
pip install vizseq==0.1.15
pip install nltk==3.6.4
pip install sacrebleu==1.5.1

