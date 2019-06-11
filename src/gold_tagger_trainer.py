from util.utils import make_vocabs
from lib.GoldTagger import GoldTagger
from lib.PerceptronTrainer import PerceptronTrainer
from constants.globals import PAD
from collections import defaultdict
import random



def train_perceptron(train_data, n_epochs=1, encoded = True):
    train_data = list(train_data)  # because we will shuffle in-place
    vocab_words, vocab_tags = make_vocabs(train_data)

    tagger = GoldTagger(vocab_words, vocab_tags, encoded)
    if encoded:
        trainer = PerceptronTrainer(tagger.model)
    else:
        trainer = PerceptronTrainer(tagger.model, vocab_tags)

    #MOST FREQ
    pre_sufix = {PAD:0}
    for sentence in train_data:
      for i, (w, tag) in enumerate(sentence):
        if w[:1] not in pre_sufix:
          pre_sufix[w[:1]] = len(pre_sufix)
        if w[:2] not in pre_sufix:
          pre_sufix[w[:2]] = len(pre_sufix)
        if w[:3] not in pre_sufix:
          pre_sufix[w[:3]] = len(pre_sufix)
        if w[:4] not in pre_sufix:
          pre_sufix[w[:4]] = len(pre_sufix)
        if w[-1:] not in pre_sufix:
          pre_sufix[w[-1:]] = len(pre_sufix)
        if w[-2:] not in pre_sufix:
          pre_sufix[w[-2:]] = len(pre_sufix)
        if w[-3:] not in pre_sufix:
          pre_sufix[w[-3:]] = len(pre_sufix)
        if w[-4:] not in pre_sufix:
          pre_sufix[w[-4:]] = len(pre_sufix)
       
    tagger.pre_sufix = pre_sufix 
    for _ in range(n_epochs):
        random.shuffle(train_data)
        for i, sentence in enumerate(train_data):
            words, gold_tags = zip(*sentence)
            pred_tags = []
            for i, gold_tag in enumerate(gold_tags):
                features = tagger.featurize(words, i, pred_tags, sentence[i+1][1] if i + 1 < len(sentence) else PAD)
                trainer.update(features, gold_tag)
                pred_tags.append(gold_tag)
    trainer.finalize()
    return tagger
