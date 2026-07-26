# Semana 2 — Embeddings y modelo de lenguaje

Un modelo de lenguaje a nivel de caracteres (estilo makemore) construido con PyTorch,
para entender cómo una red trabaja con lenguaje: embeddings, softmax, cross-entropy,
entrenamiento e inferencia (generación de nombres).

## Qué hace

`embeddings.py` construye, de punta a punta:

- **Vocabulario + tabla de embeddings** — cada token (letra) se representa como un vector
- **Dataset** — pares de (contexto → siguiente token)
- **Forward** — embeddings → logits → softmax (probabilidades)
- **Loss** — cross-entropy (mira la probabilidad de la letra correcta)
- **Entrenamiento** — el mismo bucle de la Semana 1 (zero-grad → backward → update)
- **Inferencia** — genera nombres nuevos con sampling + autoregresión
- **Visualización** — grafica los embeddings 2D (`embeddings.png`)

## Cómo correrlo

Necesitas PyTorch y matplotlib:

```bash
pip install torch matplotlib
python3 embeddings.py
```

Verás el loss bajar durante el entrenamiento, y se genera `embeddings.png`
con las letras ubicadas en el plano (posiciones aprendidas por la red).

## Créditos

Basado en la serie [makemore](https://github.com/karpathy/makemore) de Andrej Karpathy
y su curso [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

Escribí sobre cómo lo entendí (y en qué me confundí) en mi blog:
[Semana 2: embeddings — cómo una red entiende letras](https://devedux.github.io/blog/semana-2-embeddings).
