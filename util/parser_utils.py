from lib.Parser import Parser
from constants.globals import UNK
from collections import defaultdict
import random

def uas(parser, gold_data):
    total = 0
    correct = 0
    for i, sentence in enumerate(gold_data):
        prog = int(i/len(gold_data)*50)
        words, tags, gold_heads = zip(*sentence)
        pred_heads = parser.predict(words, tags)
        for i, h in enumerate(pred_heads[1:], 1):
            total +=1
            if gold_heads[i] == h:
                correct += 1
    return correct/total

def oracle_moves(gold_heads):
    dependents = defaultdict(int)
    for head in gold_heads:
        dependents[head] += 1

    m = None
    config = Parser.initial_config(len(gold_heads))
    i, stack, heads = config
    while not Parser.is_final_config(config):
        valid_moves = Parser.valid_moves(config)
        second_topmost = stack[-2] if len(stack) >=2 else None
        topmost = stack[-1] if len(stack) >= 1 else None

        if Parser.LA in valid_moves and gold_heads[second_topmost] == topmost and dependents[second_topmost] == 0:
            m = Parser.LA
            dependents[topmost] -= 1
        elif Parser.RA in valid_moves and gold_heads[topmost] == second_topmost and dependents[topmost] == 0:
            m = Parser.RA
            dependents[second_topmost] -= 1
        elif Parser.SH in valid_moves:
            m = Parser.SH
        else:
            raise "Oracle failed, no moves valid"
        yield config, m
        config = Parser.next_config(config, m)

def samples(gold_data, parser, n_epochs=1):
    pairs = []
    random.shuffle(gold_data)
    for sentence in gold_data:
        words, tags, indices = zip(*sentence)
        gold_moves = oracle_moves(indices)
        for m in gold_moves:
            config, move = m
            pairs.append((parser.featurize(words,tags,config),move))
    return pairs

def output(model, dev_data, p_dev_data, vocab_tags):
    result = ""
    for i in range(len(dev_data)):
        words, tags, _ = zip(* p_dev_data[i])
        pred_heads = model.predict(words, tags)
        for j, h in enumerate(pred_heads[1:], 1):
          dev_data[i][j-1][6] = h        # -1 since we skip the first head(root) added in read_data
          result += "\t".join(map(str, dev_data[i][j-1]))
        result += '\n'
    return result

def encode(gold_data, vocab_words, vocab_tags):
    encoded_data = []
    for tagged_sentence in gold_data:
        encoded_sentence = []
        for word, gold_tag, head in tagged_sentence:
            word = word if word in vocab_words else UNK
            encoded_word = vocab_words[word]
            encoded_tag = vocab_tags[gold_tag]
            encoded_sentence.append((encoded_word, encoded_tag, head))
        encoded_data.append(encoded_sentence)
    return encoded_data
