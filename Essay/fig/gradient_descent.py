import torch
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

a = 1
b = 0
c = -2
d = 0

theta = torch.tensor(1.8, requires_grad=True)
n_iter = 10
alpha = 0.05

x = torch.linspace(0, 2, 100)


def loss(x):
    y = a * x ** 3 + b * x ** 2 + c * x + d
    return y


def gradient_descent(x, alpha, n_iter):
    x_values = []
    y_values = []
    for _ in range(n_iter):
        x_values.append(x.clone())
        y = loss(x)
        y_values.append(y)
        y.backward()
        x.data = x - alpha * x.grad
        x.grad.zero_()
    return x_values, y_values


if __name__ == "__main__":
    plt.style.use("ggplot")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.2, 4)
    ax.plot(x, loss(x))
    ax.plot(*gradient_descent(theta, alpha, n_iter), "k.")
    # ax.set_yscale("log")
    ax.set_title("Gradient Descent")
    ax.grid(True)
    tikzplotlib.save("gradient.tex")
    plt.savefig("gradient.png")
