from util.utils import read_data, filter_data, make_vocabs
from util.parser_utils import uas, output
from src.feature_parser_trainer import train_perceptron
from constants.globals import EPOCHS
import bz2

def eval_feature_parser(train_file, dev_file):
    with bz2.open(train_file, 'rt', encoding="utf-8") as source:
        train_data = list(read_data(source))

    with bz2.open(dev_file, 'rt', encoding="utf-8") as source:
        dev_data = list(read_data(source))

    train_data = filter_data(train_data, [1, 3, 6])
    dev_data = filter_data(dev_data, [1, 3, 6])
    vocab_words, vocab_tags = make_vocabs(train_data)

    perceptron_parser = train_perceptron(train_data, n_epochs=EPOCHS)
    print("UAS score for feature engineered perceptron:")
    print("{:.4f}".format(uas(perceptron_parser, dev_data)))
    print()

if __name__ == '__main__':
    eval_feature_parser("files/train.conllu.bz2", "files/dev.conllu.bz2")
