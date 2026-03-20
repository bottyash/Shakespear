import random
from collections import Counter
#Cleaning and tokenizing Shakespeare text
def tokenize(text):
    words = [word.lower() for word in text.split() if len(word) > 2]
    return words

def build_vocabulary(text): #Building vocab
    words = tokenize(text)
    vocab = list(set(words))
    return vocab

def create_model(vocab): #a tiny nn model of 2 layer
    model = {}
    model['vocab'] = vocab
    model['word_count'] = Counter(vocab)
    model['most_common'] = model['word_count'].most_common(100)
    return model

def generate_response(model, seed_word="the"): #Generate responses
    response = [seed_word] # Start with seed word
    for _ in range(15): # Generate 15 words of text
        next_word = random.choice(model['most_common']) # Get next word from model in shakespear style
        response.append(next_word)
    
    return " ".join(response).capitalize()

def generate_response(model, seed_word="the"):
    response = [seed_word]
    for _ in range(15):
        next_word = random.choice(model['most_common'])
        response.append(next_word)
    
    return " ".join(response).capitalize()

def cli_chat():
    print("Welcome to the Shakespear Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Good bye")
            break
        else:
            model = create_model(text) # Generate response
            response = generate_response(model, seed_word=user_input)
            print(f"\nShakespeare: {response}")

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        text = f.read()
    
    cli_chat()