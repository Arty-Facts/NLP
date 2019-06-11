from util.utils import make_vocabs
from lib.PerceptronTagger import PerceptronTagger
from lib.PerceptronTrainer import PerceptronTrainer
import random

def train_perceptron(train_data, n_epochs=1, encoded = True):
    train_data = list(train_data)  # because we will shuffle in-place
    vocab_words, vocab_tags = make_vocabs(train_data)

    tagger = PerceptronTagger(vocab_words, vocab_tags, encoded)
    if encoded:
        trainer = PerceptronTrainer(tagger.model)
    else:
        trainer = PerceptronTrainer(tagger.model, vocab_tags)

    for _ in range(n_epochs):
        random.shuffle(train_data)
        for i, sentence in enumerate(train_data):
            words, gold_tags = zip(*sentence)
            pred_tags = []
            for i, gold_tag in enumerate(gold_tags):
                features = tagger.featurize(words, i, pred_tags)
                trainer.update(features, gold_tag)
                pred_tags.append(gold_tag)
    trainer.finalize()
    return tagger
