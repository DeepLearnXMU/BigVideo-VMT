from flickr30k_entities.flickr30k_entities_utils import *
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
import os

nlp = StanfordCoreNLP(r'../stanford-corenlp-4.3.2')#, lang='de')

def get_sentence_list():
    sentence_list = []
    with open('train_val_test2016.en','r') as f:
        for l in f:
            sentence_list.append(l.strip())
    return sentence_list

def filter_EscapeString(l):
    l = l.replace('&apos;', '\'')
    l = l.replace("&amp;", "&")
    l = l.replace("& amp ;", '&')
    l = l.replace("&quot;", '"')
    return l

def get_name_list():
    name_list = []
    with open('train_val_test2016.txt','r') as f:
        for i in f:
            name_list.append(i.split('.')[0])   
    return name_list

def fix_post_tag(phrase_pos_tag, phrase):
    tmp = []
    tmp_idx = 0
    words = phrase.split()
    for idx, i in enumerate(words):
        if i == phrase_pos_tag[tmp_idx][0]:
            tmp.append(phrase_pos_tag[tmp_idx])
            tmp_idx += 1
        else:
            str1 = phrase_pos_tag[tmp_idx][0]
            tmp_idx += 1
            while str1 != i:
                str1 += phrase_pos_tag[tmp_idx][0]
                tmp_idx += 1
            tmp.append((i, 'UNK'))
    return tmp

def write_dict(filename, dic):
    out = open(filename, 'w', encoding='utf-8')
    t = sorted(dic.items(), key=lambda item:item[1], reverse=True)
    for i in t:
        out.write(i[0] + ' ' + str(i[1]) + '\n')
    out.close()

def matching(entity_sentence, origin_sentence):
    res = dict()
    if entity_sentence == origin_sentence or entity_sentence == origin_sentence.replace('&apos;', '\'').replace("&amp;", "&").replace("&quot;", '"'):
        for i in range(len(entity_sentence.split())):
            res[i] = i
        return res
        
    pos = -1
    entity_sentence = entity_sentence.split()
    origin_sentence = origin_sentence.split()
    for idx, i in enumerate(entity_sentence):
        try:
            pos = origin_sentence.index(i, pos+1)
            res[idx] = pos
        except:
            pass
    return res

if __name__ == "__main__":
    if not os.path.exists('data/masking/data'):
        os.mkdir('data/masking/data')
    out_noun = open('data/masking/data/multi30k.noun.position', 'w', encoding='utf-8')
    out_nouns = open('data/masking/data/multi30k.nouns.position', 'w', encoding='utf-8')

    noun = defaultdict(int)
    nouns = defaultdict(int)

    name_list = get_name_list()
    sentence_list = get_sentence_list()

    for index in range(len(name_list)):
        image = name_list[index]
        origin_sentence = sentence_list[index]
        sentence = filter_EscapeString(origin_sentence)

        # a list
        x = get_sentence_data('flickr30k_entities/Sentences/'+image.split('.')[0]+'.txt')

        for j in x:
            entity_sentence = j['sentence'].replace('”','"').replace('`','\'').lower()
            if entity_sentence.replace('"','').replace(' ','') == sentence.replace('"','').replace(' ',''):
                
                matching_dict = matching(entity_sentence, origin_sentence)
                flag1 = flag2 = False
                for t in j['phrases']:
                    phrase = t['phrase'].lower()
                    begin_position = t['first_word_index']  # start from 0
                    # if 'people' in t['phrase_type']:
                    try:
                        phrase_pos_tag = nlp.pos_tag(phrase)
                        if len(phrase_pos_tag) > len(phrase.split()):
                            phrase_pos_tag = fix_post_tag(phrase_pos_tag, phrase)
                    except:
                        continue
                    assert len(phrase_pos_tag) == len(phrase.split())

                    for idx, pos_tag in enumerate(phrase_pos_tag):
                        if pos_tag[1] == 'NN':
                            entity_sentence_position = begin_position + idx
                            if entity_sentence_position in matching_dict.keys():    # this word not in multi30k's text
                                origin_sentence_position = matching_dict[entity_sentence_position]
                                noun[pos_tag[0]] += 1
                                out_noun.write(str(origin_sentence_position)+' ')
                                flag1 = True
                                # a = entity_sentence.split()[entity_sentence_position]
                                # b = origin_sentence.split()[origin_sentence_position]
                                # if a != b:
                                #     print(phrase_pos_tag)
                                #     print(matching_dict)
                                #     print(pos_tag[0])
                                #     print(entity_sentence)
                                #     print(origin_sentence)
                                #     exit()
                                        
                        elif pos_tag[1] == 'NNS':
                            entity_sentence_position = begin_position + idx
                            if entity_sentence_position in matching_dict.keys():
                                origin_sentence_position = matching_dict[entity_sentence_position]
                                nouns[pos_tag[0]] += 1
                                out_nouns.write(str(origin_sentence_position)+' ')
                                flag2 = True
                                # a = entity_sentence.split()[entity_sentence_position]
                                # b = origin_sentence.split()[origin_sentence_position]
                                # if a != b:
                                #     print(pos_tag[0])
                                #     print(entity_sentence)
                                #     print(origin_sentence)  
                if flag1 == False:
                    out_noun.write('-1')
                if flag2 == False:
                    out_nouns.write('-1')
                out_noun.write('\n')
                out_nouns.write('\n')  
                break

    # test2017 & testcoco
    with open('data/multi30k/multi30k.en', 'r', encoding='utf-8') as f:
        for idx, l in enumerate(f):
            if idx >= 31014:
                sentence = l.strip()            
                _l = nlp.pos_tag(sentence)
                if len(_l) > len(sentence.split()):
                    _l = fix_post_tag(_l, sentence)
                assert len(_l) == len(sentence.split()), 'error'

                flag1 = False
                flag2 = False
                for idx, i in enumerate(_l):
                    # 去掉动名词
                    if i[-1] == 'NN' and i[0][-3:] != 'ing' and len(i[0]) > 1:
                        out_noun.write(str(idx)+' ')
                        flag1 = True
                        noun[i[0]] += 1
                    elif i[-1] == 'NNS' and len(i[0]) > 1:
                        out_nouns.write(str(idx)+' ')
                        flag2 = True
                        nouns[i[0]] += 1

                if flag1 == False:
                    out_noun.write(str(-1))
                if flag2 == False:
                    out_nouns.write(str(-1))
                out_noun.write('\n')
                out_nouns.write('\n')

    write_dict('data/masking/data/noun.en', noun)
    write_dict('data/masking/data/nouns.en', nouns)
