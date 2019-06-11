from lib.Parser import Parser
from lib.Linear import Linear
from constants.globals import PAD
import IPython

class PerceptronFeatureParser(Parser):

    def __init__(self, vocab_words, vocab_tags):
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        self.model = Linear(len(vocab_tags))
    
    def __calc_dist(self, dist):
        if dist < 6:
            return 1
        if 6 <= dist <= 11:
            return 2
        return 3

    def featurize(self, words, tags, config):
        ENC_PAD = self.vocab_words[PAD]
        i = config[0]
        stack = config[1]
        heads = config[2]

        w_next = ENC_PAD if i == len(words) else words[i]
        w_top = ENC_PAD if len(stack) == 0 else words[stack[-1]]
        w_sec = ENC_PAD if len(stack) < 2 else words[stack[-2]]
        t_next = ENC_PAD if i == len(words) else tags[i]
        t_top = ENC_PAD if len(stack) == 0 else tags[stack[-1]]
        t_sec = ENC_PAD if len(stack) < 2 else tags[stack[-2]]
        
        valency = 0
        if len(stack) > 0:
            for head in heads:
                if head == stack[-1]:
                    valency += 1

        return [(0, w_next),
            (1, w_top),
            (2, w_sec),
            (3, t_next),
            (4, t_top), 
            (5, t_sec),
            (6, w_top, w_sec),
            (7, t_top, t_sec),
            (8, t_top, t_next),
            (9, w_top, t_top, w_sec, t_sec),
            (10, valency)]
        


    def predict(self, words, tags):
        parser = Parser()

        # 1. Start in the initial configuration for the input sentence.
        config = parser.initial_config(len(words))

        # 2. As long as there are valid moves, ask the averaged perceptron for the next move to take.

        while len(self.valid_moves(config)) != 0:
            features = self.featurize(words, tags, config)
            output_vector = self.model.forward(features)
            move = max(output_vector, key=output_vector.get)
            config = self.next_config(config, move)

        # 3. Return the list of heads associated with the final configuration.
        return config[2]