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

def generate_shakespeare_response(model, seed_word="the"):
    response = [seed_word]
    for _ in range(5):
        next_word = random.choice(model['most_common'])
        response.append(next_word)
    
    return " ".join(response).capitalize()

def generate_shakespeare_response(model, seed_word="the"):
    response = [seed_word]
    for _ in range(15):
        next_word = random.choice(model['most_common'])
        response.append(next_word)
    
    return " ".join(response).capitalize()

def shakespeare_chat():
    print("Welcome to the Shakespear Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Good bye")
            break
        else:
            model = create_shakespeare_model(shakespeare_text)
            response = generate_shakespeare_response(model, seed_word=user_input)
            print(f"\nShakespeare: {response}")

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        shakespeare_text = f.read()
    
    shakespeare_chat()