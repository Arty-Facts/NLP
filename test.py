from lib import Eisner
from src import eisner_parser_trainer
from util.utils import read_data, filter_data, make_vocabs
from util.parser_utils import uas, output
import bz2

with bz2.open("files/train.conllu.bz2", 'rt') as source:
    train_data = list(read_data(source))

with bz2.open("files/dev.conllu.bz2", 'rt') as source:
    dev_data = list(read_data(source))

train_data = filter_data(train_data, [1, 3, 6])
dev_data = filter_data(dev_data, [1, 3, 6])

vocab_words, vocab_tags = make_vocabs(train_data)

#eisner_parser_trainer.train_eisner(train_data)
eisner = Eisner.Eisner(vocab_words)
tree = eisner.build_dependency_tree(train_data[0])
print(tree)