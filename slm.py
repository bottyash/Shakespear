import random
from collections import Counter

def tokenize_shakespeare(text):
    words = [word.lower() for word in text.split() if len(word) > 2]
    return words

def build_vocabulary(text):
    words = tokenize_shakespeare(text)
    vocab = list(set(words))
    return vocab

def create_shakespeare_model(vocab):
    model = {}
    model['vocab'] = vocab
    model['word_count'] = Counter(vocab)
    model['most_common'] = model['word_count'].most_common(100)
    return model