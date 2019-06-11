from util.utils import read_data, filter_data, make_vocabs
from util.tagger_utils import output, encode
from src.gold_tagger_trainer import train_perceptron
from constants.globals import EPOCHS

import bz2
def accuracy(tagger, gold_data):
    correct = 0
    count = 0
    for sentence in gold_data:
        words = [pair[0] for pair in sentence]
        gold_tags = [pair[1] for pair in sentence]
        pred_tags = tagger.predict(words, gold_tags)
        for i in range (len(words)):
            if(pred_tags[i] == gold_tags[i]):
                correct += 1
        count += len(words)
    return correct/count

def eval_gold_tagger(train_file, dev_file):
    with bz2.open(train_file, 'rt', encoding="utf-8") as source:
        train_data = list(read_data(source))

    with bz2.open(dev_file, 'rt', encoding="utf-8") as source:
        dev_data = list(read_data(source))

    train_data = filter_data(train_data, [1, 3])
    dev_data = filter_data(dev_data, [1, 3])

    perceptron_gold_tagger = train_perceptron(train_data, EPOCHS, False)
    print("Tagger accuracy for gold perceptron:")
    print("{:.4f}".format(accuracy(perceptron_gold_tagger, dev_data)))
    print()


    # output perceptron results, can be printed or sent to file
    # output(perceptron_tagger, dev_data, t_encoded_dev_data, vocab_tags)
if __name__ == '__main__':
    eval_gold_tagger("files/train.conllu.bz2", "files/dev.conllu.bz2")
