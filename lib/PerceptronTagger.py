from lib.Tagger import Tagger
from lib.Linear import Linear
from constants.globals import PAD

class PerceptronTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags, encoded = True):
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        if encoded:
            self.model = Linear(len(vocab_tags))
        else:
            self.model = Linear(len(vocab_tags), vocab_tags)

    def featurize(self, words, i, pred_tags):
        ENC_PAD = self.vocab_words[PAD]
        current_word = words[i]
        prev_word = ENC_PAD if i == 0 else words[i - 1]
        next_word = ENC_PAD if i == len(words) - 1 else words[i + 1]
        tag = ENC_PAD if i == 0 else pred_tags[i - 1]
        return [(0, current_word), (1, prev_word), (2, next_word), (3, tag)]

    def predict(self, words):
        pred_tags = []
        for i in range(len(words)):
            features = self.featurize(words, i, pred_tags)
            output_vector = self.model.forward(features)
            tag = max(output_vector, key=output_vector.get)
            pred_tags.append(tag)
        return pred_tags
