from util.utils import read_data, filter_data, make_vocabs
from util.parser_utils import uas, output
from src.perceptron_parser_trainer import train_perceptron
from src.neural_parser_trainer import train_neural
from constants.globals import EPOCHS
import bz2

def eval_parser(train_file, dev_file):
    with bz2.open(train_file, 'rt', encoding="utf-8") as source:
        train_data = list(read_data(source))

    with bz2.open(dev_file, 'rt', encoding="utf-8") as source:
        dev_data = list(read_data(source))

    train_data = filter_data(train_data, [1, 3, 6])
    dev_data = filter_data(dev_data, [1, 3, 6])
    vocab_words, vocab_tags = make_vocabs(train_data)

    perceptron_parser = train_perceptron(train_data, n_epochs=EPOCHS)
    print("UAS score for perceptron:")
    print("{:.4f}".format(uas(perceptron_parser, dev_data)))
    print()
    # L4 read gives: ~0.6698
    # Our read gives: ~0.6643

    neural_parser = train_neural(train_data, n_epochs=EPOCHS)
    print("UAS score for neural:")
    print("{:.4f}".format(uas(neural_parser, dev_data)))
    print()
    # L4 read gives ~0.6824 - 0.7010
    # Our read gives ~0.6948

    # output perceptron results, can be printed or sent to file
    # output(perceptron_parser, dev_data, dev_data, vocab_tags)
    # output(neural_parser, dev_data, dev_data, vocab_tags)
if __name__ == '__main__':
    eval_parser("files/train.conllu.bz2", "files/dev.conllu.bz2")
