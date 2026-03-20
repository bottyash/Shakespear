import random
from collections import Counter

def tokenize_shakespeare(text):
    words = [word.lower() for word in text.split() if len(word) > 2]
    return words

def build_vocabulary(text):
    words = tokenize_shakespeare(text)
    vocab = list(set(words))
    return vocab