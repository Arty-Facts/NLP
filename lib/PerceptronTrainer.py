from lib.Linear import Linear

class PerceptronTrainer(object):
    def __init__(self, model, vocab=None):
        if vocab == None:
            self._shadow = Linear(len(model.weight))
        else:
            self._shadow = Linear(len(model.weight), vocab)
        self.model = model
        self._counter = 1.0

    def update(self, features, gold):
        vector = self.model.forward(features)
        pred_class, _ = max(vector.items(), key=lambda x: x[1])
        if(pred_class != gold):
            self.model.bias[gold] += 1.0
            self._shadow.bias[gold] += self._counter

            self.model.bias[pred_class] -= 1.0
            self._shadow.bias[pred_class] -= self._counter

            for feat in features:
                self.model.weight[pred_class][feat] -= 1.0
                self.model.weight[gold][feat] += 1.0

                #averaging
                self._shadow.weight[pred_class][feat] -= self._counter
                self._shadow.weight[gold][feat] += self._counter


        self._counter += 1.0

    def finalize(self):
        for c, w in self.model.weight.items():
            for feat, _ in w.items():
                self.model.weight[c][feat] -= self._shadow.weight[c][feat]/self._counter
            self.model.bias[c] -= self._shadow.bias[c]/self._counter
