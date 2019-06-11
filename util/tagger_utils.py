from constants.globals import UNK

def encode(gold_data, vocab_words, vocab_tags):
    encoded_data = []
    for tagged_sentence in gold_data:
        encoded_sentence = []
        for word, gold_tag in tagged_sentence:
            word = word if word in vocab_words else UNK
            encoded_word = vocab_words[word]
            encoded_tag = vocab_tags[gold_tag]
            encoded_sentence.append((encoded_word, encoded_tag))
        encoded_data.append(encoded_sentence)
    return encoded_data

def output(model, dev_data, t_dev_data, vocab_tags):
    result = ""
    for i in range(len(dev_data)):
        words = [pair[0] for pair in t_dev_data[i]]
        pred_tags = model.predict(words)
        tags = []
        for tag_id in pred_tags:
          tag = next(key for key, value in vocab_tags.items() if value == tag_id)
          tags.append(tag)

        for j in range(len(words)):
          dev_data[i][j][3] = tags[j]
          result += "\t".join(map(str, dev_data[i][j]))
        result += '\n'
    return result

def accuracy(tagger, gold_data):
    correct = 0
    count = 0
    for sentence in gold_data:
        words = [pair[0] for pair in sentence]
        pred_tags = tagger.predict(words)
        gold_tags = [pair[1] for pair in sentence]
        for i in range (len(words)):
            if(pred_tags[i] == gold_tags[i]):
                correct += 1
        count += len(words)
    return correct/count