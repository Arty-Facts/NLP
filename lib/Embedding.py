import numpy as np

from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import scipy.sparse
from torch import nn
import pickle


class Embedding(nn.Embedding):  # TODO: Inherit from object

    def __init__(self, vocab, dim=100):
        # TODO: Replace the following line with your own code
        # super().__init__(vocab, ppmi_matrix, dim=dim)
        ppmi_matrix = scipy.sparse.load_npz('files/simplewiki-ppmi.npz')
            
        self.svd = TruncatedSVD(n_components=dim)
        self.vocab = vocab
        self.vectors = self.svd.fit_transform(ppmi_matrix)

    def vec(self, word):
        # TODO: Replace the following line with your own code
        # return super().vec(word)
        index = self.vocab[word]
        return self.vectors[index]

    def similarity(self, word1, word2):
        # TODO: Replace the following line with your own code
        # return super().similarity(word1, word2)
        i1 = self.vocab[word1]
        i2 = self.vocab[word2]
        return cosine(self.vectors[i1], self.vectors[i2])

    def most_similar(self, word, n=10):
        # TODO: Replace the following line with your own code
        # return super().most_similar(word, n=n)
        minimum = 999
        word_vector = self.vectors[self.vocab[word]]
        min_vectors = [(self.vectors[0], 9999, 0) for i in range(n)]
        for i in range(len(vocab)):
            v = self.vectors[i]
            distance = cosine(word_vector, v)
            if distance != 0:
                for j in range(len(min_vectors)):
                    min_v, min_dist, _ = min_vectors[j]
                    if distance < min_dist:
                        min_vectors.insert(j, (v, distance, i))
                        min_vectors.pop(n)
                        sorted(min_vectors, key=lambda x: x[1])
                        break
        return [[x for x in self.vocab.items() if x[1] == v[2]][0][0] for v in min_vectors]
