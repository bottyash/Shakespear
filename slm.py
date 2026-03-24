import random
import numpy as np
from collections import Counter
from transformer import build_vocab, load_weights, generate, init_weights, save_weights, forward, softmax
from train import train, tokenize, CTX_LEN, N_HEADS, VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM, WEIGHTS

TONE_WORDS = {
    "angry":    ["rage", "fury", "hate", "wrath", "villain", "curse", "blood", "die", "kill", "foul"],
    "sad":      ["grief", "sorrow", "weep", "mourn", "loss", "dead", "tears", "alone", "despair", "poor"],
    "romantic": ["love", "heart", "sweet", "beauty", "kiss", "dear", "gentle", "fair", "soul", "tender"],
    "question": ["why", "what", "who", "where", "when", "how", "dost", "canst", "wouldst", "art"],
    "neutral":  []
}

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

#Generate Shakespearean responses using trigrams
def generate_trigram_response(trigrams, seed_key, length):
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

#Generate Shakespearean responses (original)
def generate_shakespeare_response(model, seed_word="the"):
    # Start with Shakespear seed word
    response = [seed_word]
    # Generate 5 10 words of Shakespeare text
    for _ in range(5):
        # Get next word from model shakespear style
        next_word = random.choice(model['most_common'])
        response.append(next_word)
    return " ".join(response).capitalize()

#RAG: retrieve top N relevant lines from Shakespeare text ---
def retrieve_context(user_input, lines, top_n=3):
    user_words = set(user_input.lower().split())
    scored = []
    for line in lines:
        line_words = set(line.lower().split())
        score = len(user_words & line_words)
        if score > 0:
            scored.append((score, line.strip()))
    scored.sort(reverse=True)
    return [line for _, line in scored[:top_n]]

#Transformer response with RAG context + convo history
def generate_transformer_response(user_input, history, lines, params, w2i, i2w, length=20):
    # RAG: pull relevant context from text
    context_lines = retrieve_context(user_input, lines)
    context_str = " ".join(context_lines)

    #build prompt: history + rag context + user input
    history_str = " ".join(history[-4:])  # last 4 turns
    prompt = f"{history_str} {context_str} {user_input}".strip()
    prompt_words = tokenize(prompt)

    #encode
    prompt_ids = [w2i.get(w, 0) for w in prompt_words][-CTX_LEN:]

    # generate
    output_ids = generate(prompt_ids, params, N_HEADS, CTX_LEN, length)
    output_words = [i2w.get(i, '') for i in output_ids[len(prompt_ids):]]
    return " ".join(output_words).capitalize() + "."

def shakespeare_chat():
    print("Welcome to the Shakespearean Chatbot!")
    print("Type 'exit' to quit. Type 'help' for commands.")
    print("Training / loading transformer...\n")

    #train /load transformer
    params, w2i, i2w = train(shakespeare_text)

    #trigram setup
    words = tokenize_shakespeare(shakespeare_text)
    trigrams = build_trigram_map(words)

    #RAG: split text into lines
    lines = [l for l in shakespeare_text.split('\n') if len(l.strip()) > 10]

    # convo history
    history = []

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye!")
            break
        elif user_input == "help":
            print("Commands: 'exit' to quit, 'help' for commands")
        else:
            tone = detect_tone(user_input)
            length = response_length(user_input)

            #trigram response as seed
            seed = find_seed(user_input, trigrams, tone)
            trigram_resp = generate_trigram_response(trigrams, seed, length)

            #transformer response with RAG + history
            transformer_resp = generate_transformer_response(
                user_input, history, lines, params, w2i, i2w, length
            )

            #use transformer if it produced something meaningful, else fall back to trigram
            response = transformer_resp if len(transformer_resp.split()) > 4 else trigram_resp

            history.append(user_input)
            history.append(response)

            print(f"\nShakespeare: {response}")

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        shakespeare_text = f.read()

    shakespeare_chat()