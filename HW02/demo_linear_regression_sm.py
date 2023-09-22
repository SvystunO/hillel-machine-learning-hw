from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


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
    X = sm.add_constant(x)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Making OLS
    # Minimizing error
    model = sm.OLS(y, X)
    results = model.fit()

    print(results.summary())
    print("Parameters: ", results.params)
    print("R2: ", results.rsquared)

    pred_ols = results.get_prediction()

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(x, results.fittedvalues, "r--.", label="OLS")
    ax.legend(loc="best")

    plt.savefig("sm_regression.png")
