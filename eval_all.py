from eval_paser import eval_parser
from eval_tagger import eval_tagger
from eval_feature_tagger import eval_feature_tagger
from eval_feature_parser import eval_feature_parser
from eval_gold_tagger import eval_gold_tagger
print("Taggers:")
#Test tagger
eval_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")

#Test feature tagger
eval_feature_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")

#Test gold tagger
eval_gold_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")
print("Parsers:")
#Test parser
eval_parser("files/train.conllu.bz2", "files/dev.conllu.bz2")

#Test feature
eval_feature_parser("files/train.conllu.bz2", "files/dev.conllu.bz2")