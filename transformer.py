import numpy as np

#Vocab
def build_vocab(words, size=5000):
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(size)]
    w2i = {w: i+1 for i, w in enumerate(top)}
    w2i['<unk>'] = 0
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

from collections import Counter

def build_vocab(words, size=5000):
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(size)]
    w2i = {w: i+1 for i, w in enumerate(top)}
    w2i['<unk>'] = 0
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

def positional_encoding(seq_len, dim):# Position encode
    pe = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i]     = np.sin(pos / (10000 ** (i / dim)))
            if i + 1 < dim:
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / dim)))
    return pe

def softmax(x, axis=-1):#Softmax /activations
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def attention(Q, K, V, mask=None):
#Attention
    d = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
    if mask is not None:
        scores += mask * -1e9
    weights = softmax(scores)
    return weights @ V

def multihead_attention(x, Wq, Wk, Wv, Wo, n_heads):
    B, T, D = x.shape
    head_dim = D // n_heads
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # split heads
    def split(t):
        return t.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3).reshape(B * n_heads, T, head_dim)

    Q, K, V = split(Q), split(K), split(V)

    #causal mask
    mask = np.triu(np.ones((T, T)), k=1)[None, :, :]

    out = attention(Q, K, V, mask)
    out = out.reshape(B, n_heads, T, head_dim).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ Wo

#Layer norm
def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    std  = x.std(-1, keepdims=True)
    return g * (x - mean) / (std + eps) + b

#FFN
def ffn(x, W1, b1, W2, b2):
    return relu(x @ W1 + b1) @ W2 + b2

#Transformer block
def transformer_block(x, params, n_heads):
    Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
    g1, b1_ln = params['g1'], params['b1_ln']
    W1, b1_ff, W2, b2_ff = params['W1'], params['b1_ff'], params['W2'], params['b2_ff']
    g2, b2_ln = params['g2'], params['b2_ln']

    # attention + residual
    attn = multihead_attention(x, Wq, Wk, Wv, Wo, n_heads)
    x = layer_norm(x + attn, g1, b1_ln)

    # ffn + residual
    ff = ffn(x, W1, b1_ff, W2, b2_ff)
    x = layer_norm(x + ff, g2, b2_ln)
    return x

#Full forward pass
def forward(token_ids, params, n_heads, ctx_len):
    B, T = token_ids.shape
    vocab_size, embed_dim = params['embed'].shape

    x = params['embed'][token_ids]
    x += positional_encoding(T, embed_dim)

    for block in params['blocks']:
        x = transformer_block(x, block, n_heads)

    logits = x @ params['embed'].T
    return logits

#init weights
def init_weights(vocab_size, embed_dim, n_heads, n_layers, ffn_dim, ctx_len):
    def r(*shape):
        return np.random.randn(*shape) * 0.02

    params = {
        'embed': r(vocab_size, embed_dim),
        'blocks': []
    }

    for _ in range(n_layers):
        block = {
            'Wq': r(embed_dim, embed_dim),
            'Wk': r(embed_dim, embed_dim),
            'Wv': r(embed_dim, embed_dim),
            'Wo': r(embed_dim, embed_dim),
            'g1': np.ones((embed_dim,)),
            'b1_ln': np.zeros((embed_dim,)),
            'W1': r(embed_dim, ffn_dim),
            'b1_ff': np.zeros((ffn_dim,)),
            'W2': r(ffn_dim, embed_dim),
            'b2_ff': np.zeros((embed_dim,)),
            'g2': np.ones((embed_dim,)),
            'b2_ln': np.zeros((embed_dim,)),
        }
        params['blocks'].append(block)

    return params

#Save /load

def save_weights(params, path='weights.npz'):
    flat = {'embed': params['embed']}
    for i, block in enumerate(params['blocks']):
        for k, v in block.items():
            flat[f'block_{i}_{k}'] = v
    np.savez(path, **flat)
    print(f"Weights saved to {path}")

def load_weights(path, n_layers):
    data = np.load(path)
    params = {'embed': data['embed'], 'blocks': []}
    for i in range(n_layers):
        block = {k: data[f'block_{i}_{k}'] for k in
                 ['Wq','Wk','Wv','Wo','g1','b1_ln','W1','b1_ff','W2','b2_ff','g2','b2_ln']}
        params['blocks'].append(block)
    return params

#Generate
def generate(prompt_ids, params, n_heads, ctx_len, length, temperature=0.8):
    result = list(prompt_ids)
    for _ in range(length):
        ctx = np.array(result[-ctx_len:])[None, :]
        logits = forward(ctx, params, n_heads, ctx_len)
        logits = logits[0, -1] / temperature
        probs = softmax(logits)
        next_id = np.random.choice(len(probs), p=probs)
        result.append(next_id)
    return result