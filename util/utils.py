from constants.globals import PAD, UNK, ROOT, READ_LIM

def make_vocabs(gold_data):
    vocab_words = {PAD: 0, UNK: 1}
    vocab_tags = {PAD: 0}
    tuple_size = len(gold_data[0][0])
    if tuple_size == 2:
      for sentence in gold_data:
        for w, t in sentence:
          if w not in vocab_words:
            vocab_words[w] = len(vocab_words)
          if t not in vocab_tags:
            vocab_tags[t] = len(vocab_tags)
    elif tuple_size == 3:
      for sentence in gold_data:
        for entity in sentence:
            word, tag, _ = entity
            if word not in vocab_words:
                vocab_words[word] = len(vocab_words)

            if tag not in vocab_tags:
                vocab_tags[tag] = len(vocab_tags)
    return vocab_words, vocab_tags


def read_data(source, n_max=READ_LIM):
    sentence = []
    data = []
    for line in source:
        # ignore comments
        if line[0] != '#':
            if line != '\n':
                line = line.split('\t')
                sentence.append(line)
            else:
                data.append(sentence)
                sentence = []
    return data

def filter_data(data, indices=[i for i in range(10)]):
    head = 6
    default_sentence = []

    # special case for parser
    if head in indices and len(indices) == 3:
        default_sentence = [(ROOT,ROOT, 0)]

    filtered_data = []
    sentence = list(default_sentence)
    for s in data:
        for t in s:
            new_t = []
            for i in indices:
                if i == 6 and t[i] != '_':
                    new_t.append(int(t[i]))
                else:
                    new_t.append(t[i])
            sentence.append(tuple(new_t))
        filtered_data.append(sentence)
        sentence = list(default_sentence)
    return filtered_data
