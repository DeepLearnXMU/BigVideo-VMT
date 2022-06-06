
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu


hypos=['三名潜水员正在水下使用一台机器。', '两个人在外面从不同的树上采摘新鲜的水果。', '一个年轻的女孩在附近的附近骑着滑板车。']
refs=['三个潜水员在水下共同使用一台机器。', '两个人从外面不同的树上摘新鲜水果。', '一个小女孩在路上不停的蹬着滑板车。']


print(sacrebleu.corpus_bleu(hypos, [refs]))
print(corpus_bleu([refs], hypos))