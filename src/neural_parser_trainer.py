from util.parser_utils import samples
from util.utils import make_vocabs
from lib.NeuralParser import NeuralParser

import torch.nn.functional as F
import torch.optim as optim
import torch
import random

def batchify(train_data, batch_size, classifier, n_epochs):
    bx = []
    by = []
    for features, m in samples(train_data, classifier, n_epochs):
        bx.append(features)
        by.append(m)
        if len(by) >= batch_size:
            ans_bx = torch.stack(bx)
            ans_by = torch.tensor(by, dtype=torch.long)
            bx = []
            by = []
            yield ans_bx, ans_by

def train_neural(train_data, n_epochs=1, batch_size=300):
    train_data = list(train_data)  # because we will shuffle in-place
    vocab_words, vocab_tags = make_vocabs(train_data)
    classifier = NeuralParser(vocab_words, vocab_tags)
    optimizer = optim.Adam(classifier.model.parameters())
    for epoch in range(1, n_epochs+1):
        random.shuffle(train_data)
        for bx, by in batchify(train_data, batch_size, classifier, n_epochs):
            optimizer.zero_grad()
            output = classifier.model.forward(bx)
            loss = F.cross_entropy(output, by)
            loss.backward()
            optimizer.step()
    return classifier