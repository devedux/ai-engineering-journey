from nn import MLP
from engine import Value

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(20):
    # forward
    ypred = [n(x) for x in xs]
    loss = Value(0.0)
    for ypred_i, ys_i in zip(ypred, ys):
        loss += (ypred_i - ys_i) ** 2

    # zero grad
    for p in n.parameters():
        p.grad = 0.0

    # backward
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)

print("predicciones:", [round(y.data, 3) for y in ypred])
print("targets:     ", ys)
