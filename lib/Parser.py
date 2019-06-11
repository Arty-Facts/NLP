class Parser(object):
    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def initial_config(word_count):
        return (0, [], [0] * word_count)

    @staticmethod
    def valid_moves(config):
        valid = []
        i, stack, heads = config
        if i <= len(heads) - 1:
            valid.append(Parser.SH)
        if len(stack) >= 2:
            valid.append(Parser.LA)
            valid.append(Parser.RA)
        return valid

    @staticmethod
    def next_config(config, move):
        i, stack, heads = config
        if move == Parser.SH:
            stack.append(i)
            i += 1
        elif move == Parser.LA:
            child = stack.pop(-2)
            parent = stack[-1]
            heads[child] = parent
        elif move == Parser.RA:
            child = stack.pop(-1)
            parent = stack[-1]
            heads[child] = parent
        return (i, stack, heads)

    @staticmethod
    def is_final_config(config):
        i, stack, buffer = config
        return i == len(buffer) and len(stack) == 1

    # non-static part

    def featurize(self, words, tags, config):
        raise NotImplementedError

    def predict(self, words, tags):
        raise NotImplementedError