import os
import json

ids_dir="/home/sata/kly/videoNMT/data/raw_texts/test.ids"
hypos_dir="/home/sata/kly/fairseq_mmt/output/vatex_baseline/baseline_archtransformer_vatex_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.1_patience10/gen-test.txt.sorted"
result_path="/home/sata/kly/fairseq_mmt/output/vatex_baseline/baseline_archtransformer_vatex_tgtzh_lr0.005_wu2000_me100_seed1_gpu1_mt4096_wd0.1_patience10/"
ids=[]
hypos=[]

with open(ids_dir,encoding='utf-8') as file:
    ids=[x.rstrip() for x in file.readlines()]
with open(hypos_dir,encoding='utf-8') as file:
    hypos=[" ".join(x.rstrip().replace(" ","")) for x in file.readlines()]


assert len(ids)==len(hypos)
dc = dict(zip(ids,  hypos))

if not os.path.exists(result_path):
    os.makedirs(result_path)
with open(result_path + 'submission.json', 'w',encoding="utf-8") as fp:
    json.dump(dc, fp)