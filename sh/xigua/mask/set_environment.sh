
#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128


#sudo apt-get update
#sudo apt-get install ffmpeg libsm6 libxext6

mkdir ~/data
mkdir -p ~/data/vatex/
mkdir ~/codes
mkdir ~/data/fairseq_bin/

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/common.tar.gz /opt/tiger/common.tar.gz
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/codes/timm.tar.gz ~/codes
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/fairseq_bin/xigua.en-zh.asr ~/data/fairseq_bin/

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/fairseq_bin/xigua.en-zh.annotations_1016_asr_1109 ~/data/fairseq_bin/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/fairseq_bin/xigua.en-zh.annotations_1016_asr_1109_share ~/data/fairseq_bin/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_mlnlc/user/kangliyan/data/fairseq_bin/xigua.en-zh.annotations_1114 ~/data/fairseq_bin/

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
pip install numpy==1.22.0

cd ~/codes
tar -zxvf timm.tar.gz
cp -r timm /home/tiger/.local/lib/python3.7/site-packages
