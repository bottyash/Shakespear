import random
from collections import Counter

TONE_WORDS = {
    "angry":    ["rage", "fury", "hate", "wrath", "villain", "curse", "blood", "die", "kill", "foul"],
    "sad":      ["grief", "sorrow", "weep", "mourn", "loss", "dead", "tears", "alone", "despair", "poor"],
    "romantic": ["love", "heart", "sweet", "beauty", "kiss", "dear", "gentle", "fair", "soul", "tender"],
    "question": ["why", "what", "who", "where", "when", "how", "dost", "canst", "wouldst", "art"],
    "neutral":  []
} #some predefined tones that i recognise

#Cleaning and tokenizing Shakespeare text
def tokenize_shakespeare(text):
    words = [word.lower() for word in text.split() if len(word) > 2]
    return words

#Building vocab
def build_vocabulary(text):
    words = tokenize_shakespeare(text)
    vocab = list(set(words))
    return vocab

#a tiny model
def create_shakespeare_model(vocab):
    #simple 2 layer neural network
    model = {}
    model['vocab'] = vocab
    model['word_count'] = Counter(vocab)
    model['most_common'] = model['word_count'].most_common(100)
    return model

def build_trigram_map(words):
    trigrams = {}
    for i in range(len(words) - 2):
        key = (words[i], words[i + 1])
        if key not in trigrams:
            trigrams[key] = []
        trigrams[key].append(words[i + 2])
    return trigrams

#detect tone from user input
def detect_tone(user_input):
    words = set(user_input.lower().split())
    for tone, keywords in TONE_WORDS.items():
        if words & set(keywords):
            return tone
    if "?" in user_input:
        return "question"
    return "neutral"

#find best seed key based on tone then user words
def find_seed(user_input, trigrams, tone):
    user_words = user_input.lower().split()
    tone_keywords = TONE_WORDS.get(tone, [])
    for word in tone_keywords:
        matches = [k for k in trigrams if k[0] == word]
        if matches:
            return random.choice(matches)
    for word in user_words:
        matches = [k for k in trigrams if k[0] == word]
        if matches:
            return random.choice(matches)
    return random.choice(list(trigrams.keys()))

#vary length based on how much the user typed
def response_length(user_input):
    word_count = len(user_input.split())
    if word_count <= 3:
        return 8
    elif word_count <= 7:
        return 15
    else:
        return 22

#Generate Shakespearean responses
def generate_response(trigrams, seed_key, length):
    result = list(seed_key)
    key = seed_key
    for _ in range(length - 2):
        choices = trigrams.get(key)
        if not choices:
            break
        next_word = random.choice(choices)
        result.append(next_word)
        key = (key[1], next_word)
    return " ".join(result).capitalize() + "."

def shakespeare_chat():
    print("Welcome to the Shakespearean Chatbot!")
    print("Type 'exit' to quit. Type 'help' for commands.")

    words = tokenize_shakespeare(shakespeare_text)
    trigrams = build_trigram_map(words)

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye!")
            break
        else:
            # Generate response
            tone = detect_tone(user_input)
            seed = find_seed(user_input, trigrams, tone)
            length = response_length(user_input)
            response = generate_response(trigrams, seed, length)
            print(f"\nShakespeare: {response}")

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        shakespeare_text = f.read()
    
    shakespeare_chat()