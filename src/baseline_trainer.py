from lib.BaselineTagger import BaselineTagger

def train_baseline(train_data):
    freq = {}
    fallback = {}
    for sentence in train_data:
        for w, tag in sentence:
            if tag in fallback:
                fallback[tag] += 1
            else:
                fallback[tag] = 1

            if w in freq:
                if tag in freq[w]:
                    freq[w][tag] += 1
                else:
                    freq[w][tag] = 1
            else:
                freq[w] = {tag: 1}

    most_frequent = {}
    for w, v in freq.items():
        m = max(v.values())
        tag = list(filter(lambda t: t[1] == m, v.items()))[0][0]
        most_frequent[w] = tag

    m = max(fallback.values())
    tag = list(filter(lambda t: t[1] == m, fallback.items()))[0][0]
    return BaselineTagger(most_frequent, tag)