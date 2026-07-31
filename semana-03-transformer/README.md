# Semana 3 — Transformer y self-attention

Un GPT mini a nivel de carácter, construido con PyTorch, con self-attention
implementada a mano (sin usar `nn.MultiheadAttention` de PyTorch): Q/K/V,
escalado, máscara causal, softmax, multi-head, feedforward, residual
connections, layer norm, positional embeddings, y el ensamblaje completo en
`GPTLanguageModel`.

## Qué hace

`attention.py` construye, de punta a punta:

- **`Head`** — una cabeza de self-attention causal (Q, K, V, scaled dot-product, máscara triangular, softmax)
- **`MultiHeadAttention`** — varias cabezas independientes en paralelo, concatenadas y proyectadas
- **`FeedForward`** — MLP `C → 4C → C` con ReLU, aplicado a cada token
- **`Block`** — self-attention + feedforward, con residual connections y layer norm (pre-norm)
- **Positional embeddings** — para que el modelo sepa en qué posición está cada token
- **`GPTLanguageModel`** — todo ensamblado: embeddings + stack de `Block` + layer norm final + proyección a `vocab_size`, con `forward` (cross-entropy) y `generate` (autoregresión)
- **Entrenamiento real** — reusa el dataset de nombres de la Semana 2, con `get_batch` (ventanas aleatorias de contexto) y `torch.optim.Adam`

## Un hallazgo honesto

El loss baja mucho más que en la Semana 2 (~0.13 vs ~1.33), pero el modelo
memoriza el corpus de entrenamiento casi palabra por palabra en vez de
generalizar. Con un dataset de 5 nombres y tanto contexto (`block_size=8`),
el modelo tiene de sobra para memorizar en vez de aprender un patrón real.
Es overfitting, documentado a propósito en el blog en vez de ocultado.

## Cómo correrlo

```bash
pip install torch
python3 attention.py
```

Vas a ver las formas de cada pieza (Head, MultiHeadAttention, Block,
GPTLanguageModel) impresas mientras se prueban, y al final el progreso del
entrenamiento y el texto generado.

## Créditos

Basado en el video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
de Andrej Karpathy, parte de su serie [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

Escribí sobre cómo lo entendí (y en qué me confundí, incluyendo 2 bugs reales)
en mi blog:
[Semana 3: Attention, la pieza que le da contexto real a un modelo](https://devedux.github.io/blog/semana-3-transformer).
