import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class Head(nn.Module):
    """Una cabeza de self-attention causal."""

    def __init__(self, C, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

        # Buffer del triangulo para la mascara causal (no es un peso entrenable)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2,-1) * head_size**-0.5);
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, C, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(C, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, C)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# Toy input: 1 "oracion" de 8 tokens, cada uno representado con un vector de 32 numeros
B, T, C = 1, 8, 32
x = torch.randn(B, T, C)

head_size = 16
head = Head(C, head_size, block_size=T)
out = head(x)
print(out.shape)

mha = MultiHeadAttention(num_heads=4, C=C, head_size=8, block_size=T)
out = mha(x)
print(out.shape)

class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.net = nn.Sequential(
             nn.Linear(C, 4*C),
             nn.ReLU(),
             nn.Linear(4*C, C)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, C, num_heads, block_size):
        super().__init__()
        head_size = C // num_heads
        self.sa = MultiHeadAttention(num_heads, C, head_size, block_size)
        self.ffwd = FeedForward(C)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



block = Block(C, num_heads=4, block_size=T)
out = block(x)

print(out.shape)

vocab_size = 27
token_embedding_table = nn.Embedding(vocab_size, C)
position_embedding_table = nn.Embedding(T, C)

idx = torch.randint(0, vocab_size, (1, T))
tok_emb = token_embedding_table(idx)
pos_emb = position_embedding_table(torch.arange(T))
x_final = tok_emb + pos_emb
print(x_final.shape)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, C, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block(C, num_heads, block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPTLanguageModel(vocab_size=27, C=32, num_heads=4, num_layers=2, block_size=T)

idx_test = torch.randint(0, 27, (1, T))
targets_test = torch.randint(0, 27, (1, T))

logits, loss = model(idx_test, targets_test)
print(logits.shape, loss)

idx_start = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx_start, max_new_tokens=10, block_size=T)
print(generated)
print(generated.shape)

words = ["emma", "olivia", "ava", "isabella", "sophia"]
chars = sorted(set("".join(words)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
stoi["."] = len(stoi)
itos[stoi["."]] = "."
vocab_size = len(stoi)

text = "." + ".".join(words) + "."
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


model = GPTLanguageModel(vocab_size=vocab_size, C=32, num_heads=4, num_layers=2, block_size=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(3000):
    xb, yb = get_batch(batch_size=16, block_size=8)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)  # equivalente a poner .grad = None en cada peso
    loss.backward()
    optimizer.step()                       # equivalente a C.data += -lr * C.grad, para TODOS los pesos

    if step % 300 == 0:
        print(step, loss.item())


idx = torch.tensor([[stoi['.']]], dtype=torch.long)
generated = model.generate(idx, max_new_tokens=40, block_size=8)
decoded = ''.join(itos[i] for i in generated[0].tolist())
print(decoded)
