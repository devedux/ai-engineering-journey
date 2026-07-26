import random
import torch
import matplotlib.pyplot as plt

words = ["emma", "olivia", "ava", "isabella", "sophia"]

chars = sorted(set("".join(words)))

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for ch, i in stoi.items() }

stoi["."] = len(stoi)
itos[stoi["."]] = "."

vocab_size = len(stoi)
embedding_dim = 2


C = torch.randn(vocab_size, embedding_dim, requires_grad=True)


def embed(ch):
    return C[stoi[ch]]

X = []
Y = []

for w in words:
    chs = "." + w + "."
    for i in range(len(chs) - 1):
        current = chs[i]
        next = chs[i + 1]
        X.append(stoi[current])
        Y.append(stoi[next])


X = torch.tensor(X)
Y = torch.tensor(Y)

emb = C[X]

W = torch.randn(embedding_dim, vocab_size, requires_grad=True)
b = torch.randn(vocab_size, requires_grad=True)

# forward pass
#logits = emb @ W + b
#print(logits.shape)
#print(logits[0])

# softmax
#counts = logits.exp()
#probs = counts / counts.sum(1, keepdim=True)

# print(probs.shape)
# print(probs[0])
# print(probs[0].sum())

# cross-entropy
#loss = -probs[torch.arange(32), Y].log().mean()
#print(loss)

# bucle de entrenamiento

for k in range(10000):
    # forward
    emb = C[X]
    logits = emb @ W + b
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss =-probs[torch.arange(32), Y].log().mean()

    # zero grad
    C.grad = None
    W.grad = None
    b.grad = None

    # backward
    loss.backward()

    # update
    C.data += -0.1 * C.grad
    W.data += -0.1 * W.grad
    b.data += -0.1 * b.grad

    #print(k, loss.item())

for _ in range(5):          # genera 5 nombres
    out = []
    ix = stoi['.']          # empieza en el token de inicio '.'

    while True:
        # forward: del token actual a las probabilidades
        emb = C[torch.tensor([ix])]
        logits = emb @ W + b
        probs = logits.exp() / logits.exp().sum(1, keepdim=True)

        # muestrear la siguiente letra (pesado por la probabilidad)
        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == stoi['.']:   # si sale '.', el nombre terminó
            break
        out.append(itos[ix])

    # print(''.join(out))

plt.figure(figsize=(8, 8))
xs = C.data[:, 0]
ys = C.data[:, 1]

plt.scatter(xs, ys, s=200)

for i in range(len(itos)):
    plt.text(xs[i], ys[i], itos[i], ha="center", va="center", color="white", fontsize=12)

plt.grid(True)
plt.title('Embeddings de las letras (2D)')
plt.savefig('embeddings.png')   # guarda la imagen
print('Gráfico guardado en embeddings.png')
