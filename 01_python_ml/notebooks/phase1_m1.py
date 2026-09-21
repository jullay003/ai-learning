import numpy as np

#1. training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([40, 50, 60, 70, 80], dtype=float)

#2. Model Parameter
w = 0.0
b = 0.0
learning_rate = 0.01

#3. Training
for step in range(1000):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)

    gradient_w = (2 / len(X)) * np.sum((y_pred - y) * X)
    gradient_b = (2 / len(X)) * np.sum(y_pred - y)

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    if step % 100 == 0:
        print(
            f"Step {step}:",
            f"loss={loss:.4f}",
            f"w={w:.4f}",
            f"b={b:.4f}"
        )

y_pred = w * X + b

print("\nFinal paramters:")
print("w =", w)
print("b =", b)

print("\nPredictions:")
print(y_pred)

print("\nActual:")
print(y)

print("\nFinal loss:")
print(np.mean((y_pred - y) ** 2))
