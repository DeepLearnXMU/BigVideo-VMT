import os
import json
import sys

ids_dir=sys.argv[1]
hypos_dir=sys.argv[2]
result_path=sys.argv[3]
name=sys.argv[4]
ids=[]
hypos=[]
print("in formating")
print(ids_dir)
print(hypos_dir)
print(result_path)
print(name)
with open(ids_dir,encoding='utf-8') as file:
    ids=[x.rstrip() for x in file.readlines()]
with open(hypos_dir,encoding='utf-8') as file:
    hypos=[" ".join(x.rstrip().replace(" ","")) for x in file.readlines()]


assert len(ids)==len(hypos)
dc = dict(zip(ids,  hypos))

if not os.path.exists(result_path):
    os.makedirs(result_path)
with open(result_path + name+'.submission.json', 'w',encoding="utf-8") as fp:
    json.dump(dc, fp)