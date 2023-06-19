import re
from tqdm import tqdm
import argparse

pat_zh = '[\u4e00-\u9fa5]+'
pat_en = '[a-zA-Z]+'



def remove_char(str, idx):
    front = str[:idx]  # up to but not including n
    back = str[idx + 1:]  # n+1 till the end of string
    return front + back



punc_arr = ',，、：:.。*&)）]】》>%！!?"”'
punc_arr_left = '(（[【《<“'


def rm_blank_punc(text):

    if text.count(' ') > 0:
        for str in punc_arr:
            bstr = ' ' + str
            text = text.replace(bstr, str)


    if text.count(' ') > 0:
        for str in punc_arr_left:
            bstr = str + ' '
            text = text.replace(bstr, str)

    return text



def rm_blank_sp(text):
    text = text.replace('\t', ' ')
    text = re.subn('\s{2,}', ' ', text)[0]
    return text.strip()



def en_detok(sent):
    sent = sent.replace(" 's", "'s").replace(" 've", "'ve").replace(" 'm", "'m").replace(" 't", "'t")
    sent = rm_blank_punc(sent)
    sent = rm_blank_sp(sent)
    return sent


def zh_detok(text):
    text = text.strip()
    index = 0
    while 1:
        index = text.find(' ', index)
        # print('\n', index )
        if index == -1: break

        l_char = text[index - 1]
        # print('l_char : ', l_char)
        if re.match(pat_zh, l_char) != None:
            text = remove_char(text, index)
            continue

        if index + 1 <= len(text):
            r_char = text[index + 1]
            # print('r_char : ', r_char)
            if re.match(pat_zh, r_char) != None:
                text = remove_char(text, index)
                continue

        index += 1

    if len(re.findall(pat_en, text)) > 1: text = en_detok(text)
    # print(text)
    text = rm_blank_punc(text)
    text = rm_blank_sp(text)
    return text


if __name__ == '__main__':
    # revise from https://blog.csdn.net/lovechris00/article/details/127436252
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    with open(input_file, "r", encoding="utf-8") as w:
        lines = [x.strip() for x in w.readlines()]

    detokenize_results = []
    for item in tqdm(lines):
        detokenize_results.append(zh_detok(item))

    print(lines[0])
    print(detokenize_results[0])
    with open(output_file, "w", encoding="utf-8") as f:
        for line in detokenize_results:
            f.write(line + "\n")




