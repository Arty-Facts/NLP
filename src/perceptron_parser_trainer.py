from lib.PerceptronParser import PerceptronParser
from lib.PerceptronTrainer import PerceptronTrainer
from util.parser_utils import samples
from util.utils import make_vocabs
import random

def train_perceptron(train_data, n_epochs=1):
    word_vocab, tag_vocab = make_vocabs(train_data)
    parser = PerceptronParser(word_vocab, tag_vocab)
    trainer = PerceptronTrainer(parser.model)

    for sample in samples(train_data, parser):
        features, gold_move = sample
        trainer.update(features, gold_move)
    trainer.finalize()
    return parser