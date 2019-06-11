import numpy as np
import torch
from lib.Embedding import Embedding
from util.parser_utils import *
import sklearn
from lib import Linear

class Eisner():

    def __init__(self, vocab):
        self.vocab = vocab
        self.model = Linear.Linear(len(vocab))

    def score(self,head_word, head_tag, tail_word, tail_tag):
        feature_vector = [(0,head_word),(1,head_tag),(2,tail_word),(3,tail_tag)]
        return self.model.forward(feature_vector)

    def build_dependency_tree(self, sentence):
        n = len(sentence)
        C = np.zeros((n,n,2,2))

        for m in range(1,n):
            for i in range(n-m):
                j = i+m

                #compute score
                head_word, head_tag, _ = sentence[i]
                tail_word, tail_tag, _ = sentence[j]
                #score1 = self.score(head_word, head_tag, tail_word, tail_tag)
                #score2 = self.score(tail_word, tail_tag, head_word, head_tag)
                score1 = abs(i-j)
                score2 = abs(j-i)

                #→ = 1, ⟵ = 0
                
                q = range(i,j)
                C[i,j,0,1] = max(C[i,q,1,0] + [x+score1 for x in C[[x+1 for x in q],j,0,0]])
                C[i,j,1,1] = max(C[i,q,1,0] + [x+score2 for x in C[[x+1 for x in q],j,0,0]])
                C[i,j,0,0] = max(C[i,q,0,1] + C[q,j,0,0])
                C[i,j,1,0] = max(C[i,q,1,0] + C[q,j,1,1])

        #print(sentence)

        return C[0,range(n),1,0]