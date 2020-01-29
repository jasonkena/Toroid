import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import opt_einsum as oe

# from apex import amp
print(torch.cuda.get_device_name(0))
torch.manual_seed(0)
device = torch.device("cuda")


def two_to_one_coordinate(size, cell):
    return cell[0] * size[0] + cell[1]


def one_to_two_coordinate(size, coord):
    i = coord // size[0]
    j = coord - (i * size[0])
    return torch.tensor([i, j])


def toroid_distance2d(size, cellA, cellB):
    wrap_dist = torch.abs(size + torch.min(cellA, cellB) - torch.max(cellA, cellB))
    no_wrap_dist = torch.abs(cellA - cellB)
    min_dist = torch.min(wrap_dist, no_wrap_dist) ** 2
    return torch.sum(min_dist)


def toroid_distance1d(size, coordA, coordB):
    return toroid_distance2d(
        size, *[one_to_two_coordinate(size, i) for i in [coordA, coordB]]
    )


def calculate_distance_grid(size):
    return torch.tensor(
        [
            [
                toroid_distance1d(size, torch.tensor([i]), torch.tensor([j]))
                for i in range(torch.prod(size))
            ]
            for j in range(torch.prod(size))
        ],
        dtype=torch.float,
    )


class Grid(nn.Module):
    def __init__(self, size):
        super(Grid, self).__init__()
        # import pdb; pdb.set_trace()
        self.grid = nn.Parameter(
            torch.rand((size ** 2).tolist()).to(device), requires_grad=True
        )
        eq = "ac,bd,cd,ab,ab->"
        self.distance_grid = calculate_distance_grid(size).to(device)
        self.triu = torch.triu(torch.ones((size ** 2).tolist())).to(device)

        self.new_size = (size ** 2).tolist()
        ops = (
            self.new_size,
            self.new_size,
            self.distance_grid,
            self.distance_grid,
            self.triu,
        )
        constants = [2, 3, 4]

        self.expr = oe.contract_expression(
            eq, *ops, constants=constants, optimize="optimal"
        )

    def forward(self):
        if torch.isnan(self.grid).any():

            class MyException(Exception):
                pass

            raise MyException("Broken")
        new_grid = softmax(self.grid)
        return self.expr(new_grid, new_grid, backend="torch")

    def ras(self, max_iter=10, iteration=0):
        torch.div(self.grid, torch.sum(a, axis=0), out=self.grid)
        a = a / (np.sum(a, axis=1)[:, np.newaxis])
        # print(a)
        if iteration >= max_iter:
            print(a)
            return a
        return f(a, max_iter, iteration + 1)

