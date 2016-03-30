import yaml

def tokenize(s):
    exclude = ['.', ',', '?', '!', '-', ')', '(', '=', ']', '[', "'", ":", ";"]
    
    s = ''.join(ch for ch in s if ch not in exclude)
    s = s.replace('\n', ' ')
    s = s.lower()
    s = " ".join(s.split())
    
    words = s.split(' ')
    return words
    
def gen_dict(paths):

    files = [ open(path, 'r') for path in paths ]
    dictionaries = [ yaml.load(dict_file) for dict_file in files ]
    map(lambda x: x.close(), files)
    dictionary = {}

    for curr_dict in dictionaries:
        for key in curr_dict:
            if key in dictionary:
                dictionary[key].extend(curr_dict[key])
            else:
                dictionary[key] = curr_dict[key]
    
    return dictionary


def tag(words, dictionary):
    tagged_words = [(word, dictionary[word]) if word in dictionary else (word, []) for word in words ]        
    return tagged_words

def value_of(tag):
    r = 0
    if 'positive' in tag:
        r = 1
    elif 'negative' in tag:
        r = -1

    if 'strong' in tag:
        r = r * 1.5
    return r


def scores(tagged):
    scores = []
    previous = None
    for word, tag in tagged:
        score = value_of(tag)
        if previous is not None:
            if 'inc' in previous:
                score *= 2.0
            elif 'dec' in previous:
                score /= 2.0
        scores.append((word, score))
        previous = tag
    return scores


def demo_analyze(s, d):
    print s + "\n\n\n\n"
    raw_input()
    
    words = tokenize(s)
    print str(words) + "\n\n\n\n"
    raw_input()

    tagged = tag(words, d)
    for tags in scores(tagged):
        print tags
    raw_input()
    
    print "\n\n\n"
    score = reduce(lambda x, y: y[1] + x, scores(tagged), 0)
    print score
    raw_input()    
    return score    

def analyze(s, d):    
    words = tokenize(s)
    tagged = tag(words, d)
    score = reduce(lambda x, y: y[1] + x, scores(tagged), 0)
    return score

