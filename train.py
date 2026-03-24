import numpy as np
import os
from collections import Counter
from transformer import (build_vocab, init_weights, forward,
                         save_weights, load_weights, softmax)

#configs
VOCAB_SIZE  = 5000
EMBED_DIM   = 64
N_HEADS     = 4
N_LAYERS    = 2
FFN_DIM     = 128
CTX_LEN     = 16
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 1e-3
WEIGHTS     = 'weights.npz'

def tokenize(text):
    return [w.lower() for w in text.split() if len(w) > 2]

def make_batches(ids, ctx_len, batch_size):
    samples = []
    for i in range(0, len(ids) - ctx_len - 1, ctx_len):
        x = ids[i:i + ctx_len]
        y = ids[i+1:i + ctx_len + 1]
        samples.append((x, y))
    np.random.shuffle(samples)
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        if len(batch) < batch_size:
            continue
        xs = np.array([s[0] for s in batch])
        ys = np.array([s[1] for s in batch])
        yield xs, ys

def cross_entropy(logits, targets):
    B, T, V = logits.shape
    probs = softmax(logits)
    loss = 0
    for b in range(B):
        for t in range(T):
            loss -= np.log(probs[b, t, targets[b, t]] + 1e-9)
    return loss / (B * T)

def grad_embed(logits, targets, params, token_ids):
    B, T, V = logits.shape
    probs = softmax(logits).reshape(B * T, V)
    tgts  = targets.reshape(B * T)
    dlogits = probs.copy()
    dlogits[np.arange(B * T), tgts] -= 1
    dlogits /= (B * T)
    #gradient for embeddings
    grad = np.zeros_like(params['embed'])
    flat_ids = token_ids.reshape(B * T)
    np.add.at(grad, flat_ids, dlogits @ params['embed'])
    return grad

def train(text):
    print("Tokenizing...")
    words = tokenize(text)
    w2i, i2w = build_vocab(words, VOCAB_SIZE)
    ids = np.array([w2i.get(w, 0) for w in words])

    if os.path.exists(WEIGHTS):
        print(f"Found {WEIGHTS}, loading weights...")
        params = load_weights(WEIGHTS, N_LAYERS)
    else:
        print("No weights found, training from scratch...")
        params = init_weights(VOCAB_SIZE + 1, EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM, CTX_LEN)

        for epoch in range(EPOCHS):
            total_loss = 0
            batches = 0
            for xs, ys in make_batches(ids, CTX_LEN, BATCH_SIZE):
                logits = forward(xs, params, N_HEADS, CTX_LEN)
                loss   = cross_entropy(logits, ys)
                total_loss += loss
                batches += 1

                #simple gradient step on embeddings only (lightweight)
                B, T, V = logits.shape
                probs = softmax(logits).reshape(B * T, V)
                tgts  = ys.reshape(B * T)
                dlogits = probs.copy()
                dlogits[np.arange(B * T), tgts] -= 1
                dlogits /= (B * T)

                flat_ids = xs.reshape(B * T)
                grad = np.zeros_like(params['embed'])
                np.add.at(grad, flat_ids, dlogits @ params['embed'])
                params['embed'] -= LR * grad

                if batches % 100 == 0:
                    print(f"  Epoch {epoch+1} | Batch {batches} | Loss {loss:.4f}")

            print(f"Epoch {epoch+1}/{EPOCHS} done | Avg loss: {total_loss/batches:.4f}")

        save_weights(params, WEIGHTS)

    return params, w2i, i2w

if __name__ == '__main__':
    with open('text.txt', 'r') as f:
        text = f.read()
    train(text)