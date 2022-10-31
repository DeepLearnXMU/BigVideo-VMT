from tqdm import trange
import re

file = "/mnt/bn/mlx-test002-lf/mlx/users/kangliyan/data/raw_texts/valid.en"
output_file = "/mnt/bn/mlx-test002-lf/mlx/users/kangliyan/data/raw_texts/valid.rc.en"

with open(file, "r", encoding="utf-8") as r:
    files = [x.strip() for x in r.readlines()]

ends_li = [".", "。", "!", "！", "?", "？"]
for i in trange(len(files)):
    sentences = re.split(r"([.。!！?？；;，,+])", files[i])

    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

    for j in range(len(sentences)):
        if j == 0:
            sentences[j] = sentences[j].capitalize()
        else:
            sentences[j] = sentences[j][1:]
            if sentences[j - 1] != "":
                if sentences[j - 1][-1] in ends_li:
                    sentences[j] = sentences[j].capitalize()

    files[i] = " ".join(sentences)

with open(output_file, "w", encoding="utf-8") as w:
    for line in files:
        w.write(line + "\n")