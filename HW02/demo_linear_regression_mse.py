from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise

mse_m_array = []
mse_b_array = []
mse_array = []

def mse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MSE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat = m * x + b
    # Get loss MSE
    mse = (np.square(y - y_hat)).mean()

    mse_m_array.append(m);
    mse_b_array.append(b);
    mse_array.append(mse);
    return mse

def generate_iteration_graph(res, title):
    num = [i for i in range(1, len(res) + 1)]
    print(num)
    plt.close('all')
    plt.plot(num, res)
    # Customize the plot
    plt.title("Line Graph")
    plt.xlabel("Iteration number")
    plt.ylabel(title)
    plt.savefig("mse_iteration_" + title + "_approximation.png")

if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Initial guess of the parameters: [2, 2] (m, b).
    # It doesn’t have to be accurate but simply reasonable.
    initial_guess = np.array([5, -3])

    # Maximizing the probability for point to be from the distribution
    results = minimize(
        mse_regression,
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

    plt.savefig("mse_regression.png")

    generate_iteration_graph(mse_m_array, 'm_param')
    generate_iteration_graph(mse_b_array, 'b_param')
    generate_iteration_graph(mse_array, 'mse')



    # Display the plot
