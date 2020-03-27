import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import opt_einsum as oe
import os

# import wandb
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# wandb.init(project="toroid", sync_tensorboard=True)

# from apex import amp
# print(torch.cuda.get_device_name(0))
# torch.manual_seed(0)
device = torch.device("cuda")
# device = torch.device("cpu")


class MyException(Exception):
    pass


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
        # Size is a torch tensor, BEFORE HyperGrid
        super(Grid, self).__init__()
        self.new_size = (size ** 2).tolist()

        # Check if exists
        filename = os.path.join("cache", str(size[0].item()) + ".pth")
        if os.path.isfile(filename):
            self.expr = torch.load(filename)
        else:
            self.distance_grid = calculate_distance_grid(size).to(device)
            eq = "ac,bd,cd,ab->"

            ops = (self.new_size, self.new_size, self.distance_grid, self.distance_grid)
            constants = [2, 3]

            self.expr = oe.contract_expression(
                eq, *ops, constants=constants, optimize="optimal"
            )
            torch.save(self.expr, filename)

        self.revealed = []

    def grid_loss(self, grid):
        return self.expr(grid, grid, backend="torch")

    def forward(self, optim, beta):
        for n_iter in range(self.new_size[0]):
            grid = torch.ones(self.new_size, device=device) / (
                self.new_size[0] - len(self.revealed)
            )
            for reveal in self.revealed:
                grid[reveal[0], :] = 0
                grid[:, reveal[1]] = 0
                grid[reveal] = 1
            grid.requires_grad = True

            for n_optim in range(optim):
                assert grid.grad is None or torch.all(grid.grad == 0)
                gridloss = self.grid_loss(grid)
                gridloss.backward()

                assert torch.all(grid.grad > 0)
                assert torch.all(grid >= 0)

                prefix = "real_" if (n_optim == 0) else ""
                global_step = n_iter * optim + n_optim

                writer.add_scalar(prefix + "grid_loss", gridloss, global_step)
                writer.add_image(prefix + "grid", grid, global_step, dataformats="HW")
                writer.add_image(
                    prefix + "filtered_grid",
                    filter_tensor(grid, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    prefix + "filtered_grid_grad",
                    filter_tensor(grid.grad, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    prefix + "grid_grad", grid.grad, global_step, dataformats="HW",
                )

                with torch.no_grad():
                    # lr
                    alpha = (1 - beta) * grid / grid.grad
                    lr = torch.min(filter_tensor(alpha, self.revealed, only=False))
                    assert lr != 0
                writer.add_scalar("lr", lr, global_step)

                for reveal in self.revealed:
                    grid.grad[reveal[0], :] = -1
                    grid.grad[:, reveal[1]] = -1

                # Gradient Descent
                grid.data = grid - lr * grid.grad
                grid.grad.zero_()

            gridloss = self.grid_loss(grid)
            gridloss.backward()
            # NOTE: maybe RAS here
            if optim == 0:
                writer.add_scalar("real_grid_loss", gridloss, n_iter)
                for reveal in self.revealed:
                    grid.grad[reveal[0], :] = float("inf")
                    grid.grad[:, reveal[1]] = float("inf")
                self.revealed.append(
                    divmod(torch.argmin(grid.grad).item(), self.new_size[0])
                )
            else:
                for reveal in self.revealed:
                    grid[reveal[0], :] = 0
                    grid[:, reveal[1]] = 0
                self.revealed.append(
                    divmod(torch.argmax(grid).item(), self.new_size[0])
                )

        # Assert that grid is already discrete
        return grid

    def discretization(self, grid):
        grid = grid.clone()
        assert torch.all(grid >= 0)
        discrete_grid = torch.zeros(self.new_size, device=device)
        while (grid >= 0).any():
            index = divmod(torch.argmax(grid).item(), self.new_size[0])
            discrete_grid[index] = 1
            grid[index[0]] = -1
            grid[:, index[1]] = -1
        return discrete_grid

    def ras(self, tensor, n_iter):
        for _ in range(n_iter):
            # Where proportion is % of original
            scale_a = torch.diag(1 / torch.sum(tensor, axis=1))
            tensor = scale_a @ self.tensor
            scale_b = torch.diag(1 / torch.sum(tensor, axis=0))
            tensor = tensor @ scale_b
        return tensor


def filter_tensor(tensor, revealed, only, fill=False, shape=None):
    # If Only: only revealed will be returned
    # If Not Only: everything but revealed
    revealed = torch.tensor(revealed)
    revealed_rows = revealed[:, 0]
    revealed_cols = revealed[:, 1]

    index_rows = torch.any(
        revealed_rows.unsqueeze(1)
        == torch.arange((shape[0] if fill else tensor.size(0))).unsqueeze(0),
        dim=0,
    ).unsqueeze(1)
    index_cols = torch.any(
        revealed_cols.unsqueeze(1)
        == torch.arange((shape[1] if fill else tensor.size(1))).unsqueeze(0),
        dim=0,
    ).unsqueeze(0)

    if not only:
        index_rows = ~index_rows
        index_cols = ~index_cols

    if not fill:
        result = tensor[index_rows & index_cols].view(len(revealed), len(revealed))
    else:
        result = torch.zeros(shape)
        result[index_rows & index_cols] = torch.flatten(tensor)
    return result


if __name__ == "__main__":
    size = torch.tensor([6, 6])
    # number of iterations
    # optim = 0 and optim = 1 exactly identical
    optim = 10
    # how much is left
    beta = 0.1
    writer.add_hparams({"size": size[0].item()}, {})
    grid = Grid(size)
    result = grid.forward(optim, beta)
