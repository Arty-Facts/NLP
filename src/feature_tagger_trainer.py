from util.utils import make_vocabs
from lib.FeatureTagger import FeatureTagger
from lib.PerceptronTrainer import PerceptronTrainer
from constants.globals import PAD
from collections import defaultdict
import random

def train_perceptron(train_data, n_epochs=1, encoded = True):
    train_data = list(train_data)  # because we will shuffle in-place
    vocab_words, vocab_tags = make_vocabs(train_data)

    tagger = FeatureTagger(vocab_words, vocab_tags, encoded)
    if encoded:
        trainer = PerceptronTrainer(tagger.model)
    else:
        trainer = PerceptronTrainer(tagger.model, vocab_tags)
    
    #MOST FREQ
    pre_sufix = {PAD:0}
    freq = defaultdict(lambda : defaultdict(int))
    fallback = defaultdict(int)
    for sentence in train_data:
      for i, (w, tag) in enumerate(sentence):
        fallback[tag] += 1
        freq[w][tag] += 1
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
        #if i + 1 != len(sentence):
        #  freq[(w,sentence[i+1][0])][tag] += 1
        if i + 1 != len(sentence):
          freq[(sentence[i-1][0]),w][tag] += 1
       
    tagger.pre_sufix = pre_sufix 
    count = 0
    for w, v in freq.items():
      m = max(v.values())
      tag = list(filter(lambda t: t[1]==m, v.items()))[0][0]
      if isinstance(w, tuple) and freq[w][tag] > 10:
        count += 1
        tagger.most_frequent[w] = tag
      elif not isinstance(w, tuple):
        tagger.most_frequent[w] = tag
    #MOST FREQ
    m = max(fallback.values())
    tag = list(filter(lambda t: t[1]==m, fallback.items()))[0][0]
    tagger.fallback = tag

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
