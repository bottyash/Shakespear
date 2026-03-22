import random
from collections import Counter

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

def build_bigram_map(words):
    pairs = {}
    for i in range(len(words) - 1):
        w = words[i]
        if w not in pairs:
            pairs[w] = []
        pairs[w].append(words[i + 1])
    return pairs

def generate_response(pairs, seed, length=12):
    if seed not in pairs:
        seed = random.choice(list(pairs.keys()))
    result = [seed]
    for _ in range(length - 1):
        choices = pairs.get(result[-1])
        if not choices:
            break
        result.append(random.choice(choices))
    return " ".join(result).capitalize() + "."

#Generate Shakespearean responses
def generate_shakespeare_response(model, seed_word="the"):
    # Start with Shakespear seed word
    response = [seed_word]
    
    # Generate 5 10 words of Shakespeare text
    for _ in range(5):
        # Get next word from model shakespear style
        next_word = random.choice(model['most_common'])
        response.append(next_word)
    
    return " ".join(response).capitalize()

def shakespeare_chat():
    print("Welcome to the Shakespearean Chatbot!")
    print("Type 'exit' to quit. Type 'help' for commands.")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye!")
            break
        else:
            # Generate response
            words = tokenize_shakespeare(shakespeare_text)
            pairs = build_bigram_map(words)
            seed = user_input.split()[0]
            response = generate_response(pairs, seed)
            print(f"\nShakespeare: {response}")

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        shakespeare_text = f.read()
    
    shakespeare_chat()
