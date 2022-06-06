import json

from tqdm import tqdm

train_file="/home/sata/kly/videoNMT/data/vatex_training_v1.0.json"
dev_file="/home/sata/kly/videoNMT/data/vatex_validation_v1.0.json"
test_file="/home/sata/kly/videoNMT/data/vatex_public_test_english_v1.1.json"
#with open('vatex_training_v1.0.json', 'r', encoding='utf-8') as file:
    #data = json.load(file)
with open(train_file, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
with open(dev_file, 'r', encoding='utf-8') as file:
    dev_data = json.load(file)
with open(test_file, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

#read
src='en'
tgt='zh'
maps = {'en':'enCap', 'zh':'chCap'}

train_sent_ids=[]
for d in train_data:
    srccap = d[maps[src]][5:]
    sent_id = [''.join((d['videoID'],'&',str(i))) for i in range(len(srccap))]
    train_sent_ids.extend(sent_id)

dev_sent_ids=[]
for d in dev_data:
    srccap = d[maps[src]][5:]
    sent_id = [''.join((d['videoID'],'&',str(i))) for i in range(len(srccap))]
    dev_sent_ids.extend(sent_id)

test_sent_ids=[]
for d in test_data:
    srccap = d[maps[src]][5:]
    sent_id = [''.join((d['videoID'],'&',str(i))) for i in range(len(srccap))]
    test_sent_ids.extend(sent_id)

train_output="/home/sata/kly/videoNMT/data/raw_texts/train"
with open(train_output+".ids", "w", encoding="utf-8") as w:
    for line in tqdm(train_sent_ids):
        w.write(line+ "\n")

dev_output="/home/sata/kly/videoNMT/data/raw_texts/dev"
with open(dev_output+".ids", "w", encoding="utf-8") as w:
    for line in tqdm(dev_sent_ids):
        w.write(line+ "\n")

test_output="/home/sata/kly/videoNMT/data/raw_texts/test"
with open(test_output+".ids", "w", encoding="utf-8") as w:
    for line in tqdm(test_sent_ids):
        w.write(line+ "\n")

# train_srccaps=[]
# train_tgtcaps=[]
# for d in train_data:
#     srccap = d[maps[src]][5:]
#     train_srccaps.extend(srccap)
#     tgtcap = d[maps[tgt]][5:]
#     train_tgtcaps.extend(tgtcap)
#
# dev_srccaps=[]
# dev_tgtcaps=[]
# for d in dev_data:
#     srccap = d[maps[src]][5:]
#     dev_srccaps.extend(srccap)
#     tgtcap = d[maps[tgt]][5:]
#     dev_tgtcaps.extend(tgtcap)
#
# test_srccaps=[]
# for d in test_data:
#     srccap = d[maps[src]][5:]
#     test_srccaps.extend(srccap)
#
# print(len(train_srccaps),len(dev_srccaps),len(test_srccaps))
# #write
# train_output="/home/sata/kly/videoNMT/data/raw_texts/train"
# with open(train_output+".en", "w", encoding="utf-8") as w:
#     for line in tqdm(train_srccaps):
#         w.write(line+ "\n")
# with open(train_output+".zh", "w", encoding="utf-8") as w:
#     for line in tqdm(train_tgtcaps):
#         w.write(line+ "\n")
#
# dev_output="/home/sata/kly/videoNMT/data/raw_texts/dev"
# with open(dev_output+".en", "w", encoding="utf-8") as w:
#     for line in tqdm(dev_srccaps):
#         w.write(line+ "\n")
# with open(dev_output+".zh", "w", encoding="utf-8") as w:
#     for line in tqdm(dev_tgtcaps):
#         w.write(line+ "\n")
#
# test_output="/home/sata/kly/videoNMT/data/raw_texts/test"
# with open(test_output+".en", "w", encoding="utf-8") as w:
#     for line in tqdm(test_srccaps):
#         w.write(line+ "\n")
