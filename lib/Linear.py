from collections import defaultdict
class Linear(object):

    def __init__(self, output_size, vocab=None):
        self.output_size = output_size
        if vocab == None:
            self.weight = {i: defaultdict(float) for i in range(output_size)}
            self.bias = {i: 0.0 for i in range(output_size)}
        else:
            self.weight = {k: defaultdict(float) for k, v in vocab.items()}
            self.bias = {k: 0.0 for k, v in vocab.items()}

    def forward(self, features):
        output_vector = defaultdict(float)
        for c, weight in self.weight.items():
            output_vector[c] = 0
            for feat in features:
                output_vector[c] += weight[feat]
        return output_vector
