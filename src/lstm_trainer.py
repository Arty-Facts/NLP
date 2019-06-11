from lib.LSTM_Tagger import LSTM_Tagger
from util.utils import make_vocabs
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random

def batchify(train_data):
    for tagged_sentence in train_data:
      words, gold_tags = zip(*tagged_sentence)
      yield torch.tensor(words, dtype=torch.long), torch.tensor(gold_tags, dtype=torch.long)

def train_ltsm(train_data, n_epochs=1, batch_size=100):
    train_data = list(train_data)
    vocab_words, vocab_tags = make_vocabs(train_data)
    model = LSTM_Tagger(50, 300, len(vocab_words), len(vocab_tags))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(n_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        random.shuffle(train_data)
        c = 0
        for sentence, tags in batchify(train_data):
            c += 1
            print(f"{c}/{len(train_data)}", end="\r")
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            tag_scores = model(sentence)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()
    return model