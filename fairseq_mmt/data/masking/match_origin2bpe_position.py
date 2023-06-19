import os

src_tgt = 'en-de'

now_path = os.getcwd()
if not os.path.exists(os.path.join(now_path, 'data', src_tgt)):
    os.mkdir(os.path.join(now_path, 'data', src_tgt))

data_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
multi30k_dir = os.path.join(data_path, 'multi30k')

count = 0
_list = []

_f = open(os.path.join(multi30k_dir, 'multi30k.en'), 'r', encoding='utf-8')
with open(os.path.join(multi30k_dir, 'multi30k-'+src_tgt+'.bpe.en'), 'r', encoding='utf-8') as f:
    for sentence_bpe, sentence in zip(f, _f):
        count += 1
        bpe =  sentence_bpe.strip().split()
        origin =  sentence.strip().split()

        dic = {}
        if len(origin) == len(bpe):
            dic = -1
            _list.append(dic)
        else:
            _str = ""
            v = 0
            for i in range(len(origin)):
                if origin[i] == bpe[i+v]:
                    dic[i] = [i+v]
                else:
                    for j in range(i+v, len(bpe)):
                        _str += bpe[j].replace('@@', '')
                        if _str == origin[i]:
                            dic[i] = [x for x in range(i+v, j+1)]
                            _str = ""
                            v = j-i
                            break
                        
            _list.append(dic)

with open(os.path.join(now_path, 'data', src_tgt, 'origin2bpe.'+src_tgt+'.match'), 'w', encoding='utf-8') as f:
    for i in _list:
        if isinstance(i, int):
           f.write(str(-1)+'\n')
        else:
           for k,v in i.items():
              f.write(str(k)+':')
              for j in v[:-1]:
                  f.write(str(j)+'-')
              f.write(str(v[-1])+' ')
           f.write('\n')

_f.close()
