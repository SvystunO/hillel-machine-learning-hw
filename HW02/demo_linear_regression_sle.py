from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # SLE is Xa = y
    # X^(-1)Xa = X^(-1)y
    # Ea = X^(-1)y
    # a = X^(-1)y
    # For more info, please refer to https://towardsdatascience.com/70820a3fbdb9

    X = np.expand_dims(x, axis=1)
    X = np.hstack((X, np.ones_like(X)))
    Xinv = np.linalg.pinv(X)
    result = np.matmul(Xinv, y)
    m = result[0]
    b = result[1]
    print("Parameters: ", m, b)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(x, m*x+b, "r--.", label="SLE")
    ax.legend(loc="best")

    plt.savefig("sle_regression.png")
