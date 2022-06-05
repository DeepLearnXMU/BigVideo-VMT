import spacy
from tqdm import tqdm,trange

train='/home/sata/kly/videoNMT/data/raw_texts/train.zh'
dev='/home/sata/kly/videoNMT/data/raw_texts/dev.zh'
output='/home/sata/kly/videoNMT/data/raw_texts/'
nlp = spacy.load("zh_core_web_sm")

with open(train,encoding='utf-8') as file:
    train=[x.rstrip() for x in file.readlines()]

with open(dev, encoding='utf-8') as file:
    dev = [x.rstrip() for x in file.readlines()]
print(train[:3])
print(dev[:3])

for i in trange(len(train)):
    train[i]=" ".join([t.text for t in  nlp(train[i])])

for i in trange(len(dev)):
    dev[i]=" ".join([t.text for t in  nlp(dev[i])])


print(train[:3])
print(dev[:3])

with open(output+"train.tok.zh", "w", encoding="utf-8") as w:
    for line in tqdm(train):
        w.write(line+ "\n")
with open(output+"dev.tok.zh", "w", encoding="utf-8") as w:
    for line in tqdm(dev):
        w.write(line+ "\n")
