# Semana 1 — Backpropagation desde cero

Un motor de autograd mínimo (estilo micrograd) construido a mano para entender
cómo se entrena una red neuronal: forward pass, backpropagation y descenso de gradiente.

## Archivos

- **`engine.py`** — la clase `Value`: envuelve un número y calcula gradientes con backpropagation
- **`nn.py`** — la red neuronal: `Neuron`, `Layer`, `MLP`
- **`train.py`** — el bucle de entrenamiento (forward → zero-grad → backward → update)

## Cómo correrlo

```bash
python3 train.py
```

Verás el loss bajar vuelta a vuelta y las predicciones acercarse a los targets.

## Créditos

Basado en el excelente [micrograd de Andrej Karpathy](https://github.com/karpathy/micrograd)
y su serie [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
de donde aprendí todo esto.

Escribí sobre cómo lo entendí (y en qué me confundí) en mi blog:
[Semana 1: cómo entendí backpropagation desde cero](https://devedux.github.io/blog/semana-1-backprop).
