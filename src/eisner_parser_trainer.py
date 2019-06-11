from lib.PerceptronParser import PerceptronParser
from lib.PerceptronTrainer import PerceptronTrainer
from util.parser_utils import samples
from util.utils import make_vocabs
import random
from lib.Eisner import Eisner


def train_eisner(train_data, n_epochs=1):
    word_vocab, tag_vocab = make_vocabs(train_data)
    parser = Eisner(word_vocab)
    trainer = PerceptronTrainer(parser.model)

    for sample in samples(train_data, parser):
        #TODO Add training loop
        
    trainer.finalize()
    return parser