import torch
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def ras(matrix):
    # Where proportion is % of original
    scale_a, scale_b = [torch.diag(torch.ones(matrix.shape[0])) for _ in range(2)]

    scale_a = torch.diag(1 / torch.sum(matrix, axis=1))
    assert not torch.isnan(scale_a).any()
    matrix = scale_a @ matrix
    assert not torch.isnan(matrix).any()
    scale_b = torch.diag(1 / torch.sum(matrix, axis=0))
    assert not torch.isnan(scale_b).any()
    matrix = matrix @ scale_b
    assert not torch.isnan(matrix).any()
    return matrix


def scaling_loss(matrix):
    row_sum = torch.sum(matrix, axis=0)
    row_column = torch.sum(matrix, axis=1)

    # This is not mean loss
    loss = torch.sum((row_sum - 1) ** 2 + (row_column - 1) ** 2)
    return loss


if __name__ == "__main__":
    num_iter = 10
    num_sample = 5
    n = 100
    results = []
    for _ in range(num_sample):
        losses = []
        matrix = torch.rand(n, n) * 2 / n
        for _ in range(num_iter):
            losses.append(scaling_loss(matrix).numpy())
            matrix = ras(matrix)
        results.append(np.array(losses))
    results = np.array(results)
    X = np.arange(num_iter)

    plt.style.use("ggplot")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, results.T)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Squared Error")
    ax.set_title("Squared Error vs Number of Iterations")
    ax.grid(True)
    tikzplotlib.save("ras.tex")

