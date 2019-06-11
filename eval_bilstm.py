from util.utils import read_data, filter_data, make_vocabs
from util.tagger_utils import output, encode, accuracy
from src.bilstm_trainer import train_biltsm
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

    lstm_tagger = train_biltsm(encoded_train_data, n_epochs=EPOCHS)
    print("Tagger accuracy for ltsm:")
    print("{:.4f}".format(accuracy(lstm_tagger, encoded_dev_data)))
    print()

if __name__ == '__main__':
    eval_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")
