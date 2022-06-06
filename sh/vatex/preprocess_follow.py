
from tqdm import tqdm
#en  lower-cased

#zh char

data_dir="/home/sata/kly/videoNMT/data/preprocess_follow/"


with open(data_dir+"train.en",encoding='utf-8') as file:
    train_en=[x.rstrip().lower() for x in file.readlines()]

with open(data_dir+"dev.en",encoding='utf-8') as file:
    dev_en=[x.rstrip().lower() for x in file.readlines()]

with open(data_dir+"test.en",encoding='utf-8') as file:
    test_en=[x.rstrip().lower() for x in file.readlines()]

with open(data_dir+"train.low.en", "w", encoding="utf-8") as w:
    for line in tqdm(train_en):
        w.write(line+ "\n")
with open(data_dir+"dev.low.en", "w", encoding="utf-8") as w:
    for line in tqdm(dev_en):
        w.write(line+ "\n")
with open(data_dir+"test.low.en", "w", encoding="utf-8") as w:
    for line in tqdm(test_en):
        w.write(line+ "\n")

with open(data_dir + "train.zh", encoding='utf-8') as file:
    train_zh = [" ".join(x.rstrip()) for x in file.readlines()]

with open(data_dir + "dev.zh", encoding='utf-8') as file:
    dev_zh = [" ".join(x.rstrip())for x in file.readlines()]


with open(data_dir+"train.char.zh", "w", encoding="utf-8") as w:
    for line in tqdm(train_zh):
        w.write(line+ "\n")
with open(data_dir+"dev.char.zh", "w", encoding="utf-8") as w:
    for line in tqdm(dev_zh):
        w.write(line+ "\n")
