from lib.NeuralTagger import NeuralTagger
from util.utils import make_vocabs
import torch.nn.functional as F
import torch.optim as optim
import torch
import random

def batchify(train_data, batch_size, classifier):
    bx = []
    by = []
    for i, tagged_sentence in enumerate(train_data):
      words, gold_tags = zip(*tagged_sentence)
      pred_tags = []
      for i, gold_tag in enumerate(gold_tags):
        bx.append(classifier.featurize(words, i, pred_tags))
        by.append(gold_tag)
        pred_tags.append(gold_tag)
        if len(by) >= batch_size:
          yield torch.stack(bx), torch.tensor(by, dtype=torch.long)
          bx , by = [] ,[]


def train_neural(train_data, n_epochs=1, batch_size=100):
    train_data = list(train_data)
    vocab_words, vocab_tags = make_vocabs(train_data)
    classifier = NeuralTagger(vocab_words, vocab_tags)
    optimizer = optim.Adam(classifier.model.parameters())
    for i in range(1, n_epochs + 1):
        random.shuffle(train_data)
        for bx, by in batchify(train_data, batch_size, classifier):
            optimizer.zero_grad()
            output = classifier.model.forward(bx)
            loss = F.cross_entropy(output, by)
            loss.backward()
            optimizer.step()
    return classifier