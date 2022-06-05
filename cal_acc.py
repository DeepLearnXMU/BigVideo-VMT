import sys
import os
from copy import deepcopy

hypo = sys.argv[1]
which_test = sys.argv[2]
task = sys.argv[3]

f1 = open(hypo, 'r', encoding='utf-8')

if which_test == 'test':
    _dir2 = '2016'
elif which_test == 'test1':
    _dir2 = '2017'
elif which_test == 'test2':
    _dir2 = 'coco'

if task == 'multi30k-en2de':
    _dir1 = 'en2de'
elif task == 'multi30k-en2fr':
    _dir1 = 'en2fr'

path = os.path.join('accurary', _dir1, _dir2)

def check1(_str, _l):
    flag = False
    for idx, i in enumerate(_l):
        if i == _str:
            flag = True
            break
    if flag:
        del _l[idx]
    return _l, flag

def check2(_str, _l):
    if '/' in _str:
        colors = _str.split('/')[:-1]
    else:
        colors = [_str]

    flag = False
    for idx, i in enumerate(_l):
        if i in colors:
            flag = True
            break
    if flag:
        del _l[idx]
    return _l, flag

if __name__ == '__main__':
    for kind in ['color', 'people']:
        restrirt_file = open(os.path.join(path, 'restrict.'+kind), 'r', encoding='utf-8')
        relax_file = open(os.path.join(path, 'relaxed.'+kind), 'r', encoding='utf-8')
        
        relaxed_num = 0
        relaxed_sum = 0
        restrict_num = 0
        restrict_sum = 0
        f1.seek(0)
        for l1, l2, l3 in zip(f1, restrirt_file, relax_file):
            src = l1.strip().split()
            restrict = l2.strip().split()
            relaxed = l3.strip().split()

            subwords = []
            for word in src:
                s = word.split('-')
                subwords.extend(s)
            tmp = deepcopy(subwords)

            #print(subwords)
            for i in restrict:
                if i == '-1':
                    break
                elif i == '-':
                    restrict_num += 1
                    restrict_sum += 1
                else:
                    restrict_sum += 1
                    subwords, t = check1(i, subwords)
                    if t:
                        restrict_num += 1

            for i in relaxed:
                if i == '-1':
                    break
                elif i == '-':
                    relaxed_num += 1
                    relaxed_sum += 1
                else:
                    relaxed_sum += 1
                    tmp, t = check2(i, tmp)
                    if t:
                        relaxed_num += 1
        
        print(kind)
        print('restrict:')
        print(str(restrict_num)+'/'+str(restrict_sum),end='=')
        print(round(restrict_num/restrict_sum * 100, 2))

        print('relaxed:')
        print(str(relaxed_num)+'/'+str(relaxed_sum),end='=')
        print(round(relaxed_num/relaxed_sum * 100, 2))

