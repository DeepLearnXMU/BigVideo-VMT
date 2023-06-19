#coding=utf-8
import sys
from tqdm import tqdm

def to_char(line1, line2):

    return " ".join((char for char in line1)), " ".join((char for char in line2))



with open(sys.argv[1], 'r', encoding='utf-8') as file1, open(sys.argv[2], 'r', encoding='utf-8') as file2, \
     open(sys.argv[1] + '.char', 'w', encoding='utf-8') as file3, open(sys.argv[2] + '.char', 'w', encoding='utf-8') as file4:

    for line1, line2 in tqdm(zip(file1, file2)):
        line1, line2 = line1.strip(), line2.strip()
        line1, line2 = to_char(line1, line2)
        file3.write(line1 + '\n')
        file4.write(line2 + '\n')