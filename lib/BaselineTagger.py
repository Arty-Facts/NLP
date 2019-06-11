from lib.Tagger import Tagger

class BaselineTagger(Tagger):

    def __init__(self, most_frequent, fallback):
        self.most_frequent = most_frequent
        self.fallback = fallback

    def predict(self, words):
        return [self.most_frequent[word] if word in self.most_frequent.keys() else self.fallback for word in words]
