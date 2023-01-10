
import os
from tqdm import tqdm

ids_dir="/mnt/bn/luyang/kangliyan/data/how2/raw_texts/valid.id"
base_dir="/mnt/bd/xigua-youtube-3/how2/video_features/VIT_patch_max32frames/valid/"

with open(ids_dir,"r",encoding="utf-8") as f:
    ids=[x.strip() for x in f.readlines()]

count=0
for item in tqdm(ids):
    v_dir = f"{base_dir}/{item}.npz"
    if not os.path.exists(v_dir):
        count+=1

print(count)