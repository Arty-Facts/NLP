from lib.Parser import Parser
from lib.Network import Network
import torch

class NeuralParser(Parser):  # TODO: Inherit from Parser

    def __init__(self, vocab_words, vocab_tags,
                 output_dim=3, hidden_dim=200,
                 word_emb_dim=50, word_features=3,
                 tag_emb_dim=10, tag_features=3):

        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

        self.word_features = word_features
        self.tag_features = tag_features

        word_types = len(vocab_words)
        tag_types = len(vocab_tags)
        embeddings = []

        for i in range(word_features):
            embeddings.append(torch.nn.Embedding(word_types, word_emb_dim))
            torch.nn.init.normal_(embeddings[i].weight, mean=0, std=0.01)
        for i in range(word_features, word_features + tag_features):
            embeddings.append(torch.nn.Embedding(tag_types, tag_emb_dim))
            torch.nn.init.normal_(embeddings[i].weight, mean=0, std=0.01)

        self.model = Network(embeddings, hidden_dim, output_dim)

    def featurize(self, words, tags, config):
        i, stack, heads = config
        if i < len(words) and words[i] not in self.vocab_words:
            self.vocab_words[words[i]] = 1
        feats = torch.zeros(self.word_features + self.tag_features, dtype=torch.long)
        feats[0] = self.vocab_words[words[i]] if i < len(words) else 0
        feats[1] = self.vocab_words[words[stack[-1]]] if len(stack) > 0 else 0
        feats[2] = self.vocab_words[words[stack[-2]]] if len(stack) > 1 else 0
        feats[3] = self.vocab_tags[tags[i]] if i < len(words) else 0
        feats[4] = self.vocab_tags[tags[stack[-1]]] if len(stack) > 0 else 0
        feats[5] = self.vocab_tags[tags[stack[-2]]] if len(stack) > 1 else 0
        return feats

    def predict(self, words, tags):
        pred_heads = []
        config = Parser.initial_config(len(words))
        while not Parser.is_final_config(config):
            valid_moves = Parser.valid_moves(config)
            features = self.featurize(words, tags, config)
            pred_moves = self.model.forward(features)
            best_m_s = [valid_moves[0], pred_moves[valid_moves[0]]]
            for m in valid_moves:
                if pred_moves[m] > best_m_s[1]:
                    best_m_s = [m, pred_moves[m]]
            config = Parser.next_config(config, best_m_s[0])
        return config[2]
