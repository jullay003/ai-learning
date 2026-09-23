import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, p):
    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

w = 0.0
b = 0.0

learning_rate = 0.1

for step in range(1000):
    z = w * X + b
    probabilities = sigmoid(z)
    loss = binary_cross_entropy(y, probabilities)
    gradient_w = np.mean((probabilities - y) * X)
    gradient_b = np.mean(probabilities - y)

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    if step % 100 == 0:
        print(
            f"Step {step}: "
            f"loss={loss:.4f}, "
            f"w={w:.4f}, "
            f"b={b:.4f}"
        )

z = w * X + b
probabilities = sigmoid(z)

predictions = (probabilities >= 0.5).astype(int)

print("\nFinal probabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)



