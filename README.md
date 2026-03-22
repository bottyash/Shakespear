# Shakespear Chatbot

A simple Python chatbot that responds using Shakespeare's work as its source text. No machine learning, no training, only Python with instant responses.

## Features
- Generates **Shakespear like phrases**
- Uses **40000 lines** of text data from *[DATA](https://www.kaggle.com/datasets/adarshpathak/shakespeare-text)*
- A small language model built to understand working of Language Models
- Generates somewhat gibberish but you get the point...
 
## Version 1 — Random Model
The original. Builds a vocabulary from the text and picks the next word randomly from the top 100 most common words. No context, no flow — purely frequency based.
 
**Key functions:** `tokenize_shakespeare`, `build_vocabulary`, `create_shakespeare_model`, `generate_shakespeare_response`

 
## Version 2 — Bigram Model
Replaced random word picking with a bigram map. Each word now points to a list of words that actually followed it in Shakespeare's text, so the response chains words together based on real co-occurrence.
 
**What changed:** added `build_bigram_map` and updated `generate_response` to walk the chain using the previous word as context.
 
**Result:** noticeably more coherent phrases compared to v1.
