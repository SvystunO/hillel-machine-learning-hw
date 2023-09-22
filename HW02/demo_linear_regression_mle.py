from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mle_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """Maximum Likelihood Estimation Regression"""
    m = guess[0]
    b = guess[1]
    sigma = guess[2]
    # Predictions
    y_hat = m * x + b
    # Compute PDF of observed values normally distributed around mean (y_hat)
    # with a standard deviation of sigma
    # Must watch: https://www.youtube.com/watch?v=Dn6b9fCIUpM
    neg_ll = -np.sum(scipy.stats.norm.logpdf(y, loc=y_hat,
                     scale=sigma))  # return negative LL
    return neg_ll


if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Initial guess of the parameters: [2, 2, 2] (m, b, sigma).
    # It doesn’t have to be accurate but simply reasonable.
    initial_guess = np.array([5, 5, 5])

    # Maximizing the probability for point to be from the distribution
    results = minimize(
        mle_regression,
        initial_guess,
        args=(x, y,),
        method="Nelder-Mead",
        options={"disp": True})
    print(results)
    print("Parameters: ", results.x)

    # Plot results
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = results.x[0] * xx + results.x[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(xx, yy, "r--.", label="MLE")
    ax.legend(loc="best")

    plt.savefig("mle_regression.png")
