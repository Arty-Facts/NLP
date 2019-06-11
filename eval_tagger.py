from util.utils import read_data, filter_data, make_vocabs
from util.tagger_utils import output, encode, accuracy
from src.perceptron_tagger_trainer import train_perceptron
from src.neural_tagger_trainer import train_neural
from constants.globals import EPOCHS
import bz2

def eval_tagger(train_file, dev_file):
    with bz2.open(train_file, 'rt', encoding="utf-8") as source:
        train_data = list(read_data(source))

    with bz2.open(dev_file, 'rt', encoding="utf-8") as source:
        dev_data = list(read_data(source))

    train_data = filter_data(train_data, [1, 3])
    dev_data = filter_data(dev_data, [1, 3])
    vocab_words, vocab_tags = make_vocabs(train_data)
    encoded_train_data = encode(train_data, vocab_words, vocab_tags)
    encoded_dev_data = encode(dev_data, vocab_words, vocab_tags)

    perceptron_tagger = train_perceptron(encoded_train_data, n_epochs=EPOCHS)
    print("Tagger accuracy for perceptron:")
    print("{:.4f}".format(accuracy(perceptron_tagger, encoded_dev_data)))
    print()
    # L3 read gives: 0.8736
    # Our read gives: 0.8736

    neural_tagger = train_neural(encoded_train_data, n_epochs=EPOCHS)
    print("Tagger accuracy for neural:")
    print("{:.4f}".format(accuracy(neural_tagger, encoded_dev_data)))
    print()
    # L3 read gives: ~0.9113
    # Our read gives: ~0.9032

    # output perceptron results, can be printed or sent to file
    # output(perceptron_tagger, dev_data, t_encoded_dev_data, vocab_tags)
    # output(neural_tagger, dev_data, t_encoded_dev_data, vocab_tags)
if __name__ == '__main__':
    eval_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")
