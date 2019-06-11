from lib.Tagger import Tagger
from lib.Network import Network
import torch

class NeuralTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags, hidden_dim=100, \
                 word_emb_dim=50, n_word_emb=3, \
                 tag_emb_dim=10, n_tag_emb=1):

        self.n_word_emb = n_word_emb
        self.n_tag_emb = n_tag_emb

        word_types = len(vocab_words)
        tag_types = len(vocab_tags)
        embeddings = []

        for i in range(n_word_emb):
            embeddings.append(torch.nn.Embedding(word_types, word_emb_dim))
            torch.nn.init.normal_(embeddings[i].weight, mean=0, std=0.01)
        for i in range(n_word_emb, n_tag_emb + n_word_emb):
            embeddings.append(torch.nn.Embedding(tag_types, tag_emb_dim))
            torch.nn.init.normal_(embeddings[i].weight, mean=0, std=0.01)

        self.model = Network(embeddings, hidden_dim, tag_types)

    def featurize(self, words, i, pred_tags):
        features = torch.zeros(self.n_word_emb + self.n_tag_emb, dtype=torch.long)
        for pos in range(self.n_word_emb):
            if pos == 0:
                features[pos] = words[i]
            elif pos == 1:
                if (i != 0):
                    features[pos] = words[i - 1]
            elif i + pos - 1 < len(words):
                features[pos] = words[i + pos - 1]
            else:
                features[pos] = 0

        for pos in range(0, self.n_tag_emb):
            if i - pos > 0:
                features[self.n_word_emb + pos] = pred_tags[(-pos - 1)]

        return features

    def predict(self, words):
        pred_tags = []
        for i in range(len(words)):
            features = self.featurize(words, i, pred_tags)
            scores = self.model.forward(features)
            pred_tag = scores.argmax()
            pred_tags.append(pred_tag)

        return pred_tags