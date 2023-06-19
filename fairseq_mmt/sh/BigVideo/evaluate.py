from tqdm import tqdm, trange
from sacrebleu.metrics import BLEU
import sacrebleu
from collections import Counter

from bleurt import score as bleurt_score
from comet import download_model, load_from_checkpoint
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="")
parser.add_argument("--disable_comet", action='store_true')
args = parser.parse_args()

checkpoint = "/BLEURT-20"

comet_model_path = download_model("wmt20-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

base_dir = args.base_dir
model = "text"
relate_score_file = "/DATA_DIR/test/test.relate_score"
video_id_file = "DATA_DIR/test/test.id"

ambiguous_tgt = []
unambiguous_tgt = []
ambiguous_output = []
unambiguous_output = []

with open(relate_score_file, "r", encoding="utf-8") as f:
    relate_score = [int(x.strip()) for x in f.readlines()]
    print(Counter(relate_score))

with open(video_id_file, "r", encoding="utf-8") as f:
    vids = [x.strip() for x in f.readlines()]

# with open(url_file,"r",encoding="utf-8") as f:
#     urls=[x.strip() for x in f.readlines()]


data_dict = {}
for idx, mask in enumerate([""]):
    print("reading...", mask)

    src = base_dir + f"{mask}/text.checkpoint_best.pt.1.0.test.src.dtk"
    tgt = base_dir + f"{mask}/text.checkpoint_best.pt.1.0.test.tgt.dtk"
    hypos = base_dir + f"{mask}/{model}.checkpoint_best.pt.1.0.test.hypo.dtk"

    with open(src, "r", encoding="utf-8") as f:
        srcs = [x.strip() for x in f.readlines()]
    with open(tgt, "r", encoding="utf-8") as f:
        tgts = [x.strip() for x in f.readlines()]
    with open(hypos, "r", encoding="utf-8") as f:
        hypos = [x.strip() for x in f.readlines()]
    print(len(srcs), len(tgts), len(hypos))

    assert len(srcs) == len(tgts) == len(hypos)
    data_dict[idx] = [srcs, tgts, hypos]



for idx, key in enumerate(data_dict.keys()):

    items = []
    ts = []
    vs = []
    tgts = []
    ambiguous_tgt = []
    unambiguous_tgt = []
    ambiguous_output = []
    unambiguous_output = []

    comet_data = []
    comet_one_data = []
    comet_zero_data = []

    for i in trange(len(data_dict[key][0])):
        src_origin = data_dict[0][0][i]
        src = data_dict[key][0][i]
        tgt = data_dict[key][1][i]
        hypo = data_dict[key][2][i]

        vs.append(hypo)
        tgts.append(tgt)

        comet_data.append({"src": src, "mt": hypo, "ref": tgt})

        if relate_score[i] == 1:
            ambiguous_output.append(hypo)
            ambiguous_tgt.append(tgt)
            comet_one_data.append({"src": src, "mt": hypo, "ref": tgt})
        elif relate_score[i] == 0:
            unambiguous_output.append(hypo)
            unambiguous_tgt.append(tgt)
            comet_zero_data.append({"src": src, "mt": hypo, "ref": tgt})

        score = sacrebleu.corpus_bleu([hypo], [[tgt]], tokenize="zh")

        items.append([src_origin, src, tgt, hypo, score.score, relate_score[i], vids[i]])


    print("======sacrebleu========")
    print("all",sacrebleu.corpus_bleu(vs, [tgts], tokenize="zh"))
    print(f"ambiguous {model}", sacrebleu.corpus_bleu(ambiguous_output, [ambiguous_tgt], tokenize="zh"))
    print(f"unambiguous {model}", sacrebleu.corpus_bleu(unambiguous_output, [unambiguous_tgt], tokenize="zh"))

    if args.disable_comet:
        print("*****over*******")
    else:
        seg_scores, sys_score = comet_model.predict(comet_data, batch_size=16, gpus=1)
        print(f"*********all {sys_score}**********")
        seg_scores, sys_score = comet_model.predict(comet_one_data, batch_size=16, gpus=1)
        print(f"*********ambiguous {sys_score}**********")
        seg_scores, sys_score = comet_model.predict(comet_zero_data, batch_size=16, gpus=1)
        print(f"*********unambiguous {sys_score}**********")

        bleurt_scorer = bleurt_score.BleurtScorer(checkpoint)
        print("=========bleurt==========")
        print("all", np.mean(bleurt_scorer.score(references=tgts, candidates=vs)))
        print("ambiguous", np.mean(bleurt_scorer.score(references=ambiguous_tgt, candidates=ambiguous_output)))
        print("unambiguous", np.mean(bleurt_scorer.score(references=unambiguous_tgt, candidates=unambiguous_output)))




