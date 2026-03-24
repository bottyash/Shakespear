# Shakespear Chatbot

A simple Python chatbot that responds using Shakespeare's work as its source text. No machine learning, no training, only Python with instant responses.

## Features
- Generates **Shakespear like phrases**
- Uses **40000 lines** of text data from *[DATA](https://www.kaggle.com/datasets/adarshpathak/shakespeare-text)*
- A small language model built to understand working of Language Models
- Generates somewhat gibberish but you get the point...

## Version 1: Random Model
The original. Builds a vocabulary from the text and picks the next word randomly from the top 100 most common words. No context, no flow — purely frequency based.
 
**Key functions:** `tokenize_shakespeare`, `build_vocabulary`, `create_shakespeare_model`, `generate_shakespeare_response`
 
## Version 2: Bigram Model
Replaced random word picking with a bigram map. Each word now points to a list of words that actually followed it in Shakespeare's text, so the response chains words together based on real co-occurrence.
 
**What changed:** added `build_bigram_map` and updated `generate_response` to walk the chain using the previous word as context.
 
**Result:** noticeably more coherent phrases compared to v1.
 
## Version 3:Trigram Model + Tone + Dynamic Length
Upgraded bigrams to trigrams (2-word context window) and added tone detection and dynamic response length.
 
**What changed:**
- `build_trigram_map` — keys are word pairs `(w1, w2)` instead of single words
- `find_seed` — scans user input for tone keywords first, then falls back to user's own words to pick the best starting pair
- `detect_tone` — classifies input as angry, sad, romantic, question, or neutral
- `response_length` — short inputs get short replies, longer inputs get longer ones
 
**Result:** responses feel contextually steered by what the user typed, not just random walks through the text.
 
## Version 4 — Transformer + RAG + Conversation History
Built a small transformer from scratch using NumPy only and wired it on top of the trigram bot.
 
**New files:**
- `transformer.py` — the model. Embeddings, positional encoding, multi-head self-attention, layer norm, FFN, generation
- `train.py` — trains on the Shakespeare data, saves weights to `weights.npz`. Loads from file on every run after the first
 
**What changed in `slm.py`:**
- `retrieve_context` — RAG: scores every line in the text by keyword overlap with user input, pulls top 3 most relevant lines as context
- `generate_transformer_response` — builds a prompt from conversation history + RAG context + user input, runs it through the transformer
- Trigram bot stays as fallback if transformer output is too short or empty
- Conversation history tracked across turns and passed into every prompt
 
**Architecture:**
- Vocab: top 5000 words
- Embedding dim: 64
- Attention heads: 4
- Layers: 2
- FFN hidden: 128
- Context window: 16 tokens
 
**First run:** trains for ~10-30 minutes and saves `weights.npz`
**Every run after:** loads weights instantly and starts chatting
 
**Result:** context-aware responses grounded in relevant Shakespeare text, with memory of the current conversation.