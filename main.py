from eval_paser import eval_parser
from eval_tagger import eval_tagger
#Test tagger
eval_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")

#Test parser
eval_parser("files/train.conllu.bz2", "files/dev.conllu.bz2")