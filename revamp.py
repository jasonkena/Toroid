import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import opt_einsum as oe
import os
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
# import wandb
from torch.utils.tensorboard import SummaryWriter

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

    def forward(self, optim, n_ras):
        for n_iter in tqdm(range(self.new_size[0])):
            # grid = (
            # torch.eye(self.new_size[0], device=device) * 0.2
            # + torch.ones(self.new_size[0], device=device) * 0.8 / self.new_size[0]
            # )
            grid = torch.ones(self.new_size, device=device) / (
                self.new_size[0] - len(self.revealed)
            )
            for reveal in self.revealed:
                grid[reveal[0], :] = 0
                grid[:, reveal[1]] = 0
                grid[reveal] = 1
            grid.requires_grad = True

            for n_optim in tqdm(range(optim), leave=False):
                assert grid.grad is None or torch.all(grid.grad == 0)
                gridloss = self.grid_loss(grid)
                gridloss.backward()

                try:
                    assert torch.all(grid.grad > 0)
                except:
                    __import__("pdb").set_trace()
                assert torch.all(grid >= 0)

                global_step = n_iter * optim + n_optim

                discrete_loss = self.grid_loss(self.discretization(grid))
                writer.add_scalar(
                    "realtime_discrete_loss", discrete_loss, global_step,
                )
                writer.add_scalar("grid_loss", gridloss, global_step)
                writer.add_image("grid", grid, global_step, dataformats="HW")
                writer.add_image(
                    "filtered_grid",
                    filter_tensor(grid, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    "filtered_grid_grad",
                    filter_tensor(grid.grad, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    "grid_grad", grid.grad, global_step, dataformats="HW",
                )

                with torch.no_grad():
                    filtered_grad = filter_tensor(grid.grad, self.revealed, only=False)
                    filtered_grid = filter_tensor(grid, self.revealed, only=False)

                    # making Gradient Zero line sum
                    filtered_grad = self.fix_grad(filtered_grad, n_ras)
                    # lr
                    # set LR for tokens increasing loss
                    alpha = (
                        filtered_grid[filtered_grad > 0]
                        / filtered_grad[filtered_grad > 0]
                    )
                    if len(alpha):
                        # NOTE: maybe use Median
                        lr = torch.mean(alpha)
                        # assert lr != 0
                    else:
                        lr = 0

                    writer.add_scalar("lr", lr, global_step)

                    filtered_grid = filtered_grid - lr * filtered_grad
                    if torch.any(filtered_grid < 0):
                        filtered_grid = self.fix_grid(filtered_grid)

                    grid.data = filter_tensor(
                        filtered_grid, self.revealed, only=False, fill=grid.data
                    )
                # Gradient Descent
                grid.grad.zero_()
                if lr == 0:
                    break

            gridloss = self.grid_loss(grid)
            gridloss.backward()
            result = grid.clone()
            # NOTE: maybe RAS here
            for reveal in self.revealed:
                grid[reveal[0], :] = 0
                grid[:, reveal[1]] = 0
            self.revealed.append(divmod(torch.argmax(grid).item(), self.new_size[0]))

        # Assert that grid is already discrete
        return result, discrete_loss

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
        assert torch.all(tensor >= 0)
        for _ in range(n_iter):
            # Where proportion is % of original
            scale_a = torch.diag(1 / torch.sum(tensor, axis=1))
            tensor = scale_a @ tensor
            scale_b = torch.diag(1 / torch.sum(tensor, axis=0))
            tensor = tensor @ scale_b
        return tensor

    def fix_grad(self, grad, n_ras):
        assert torch.all(grad > 0)
        offset = torch.tensor(1, device=device, dtype=torch.float) / grad.size(0)

        # Sum across any axis will be 0
        grad = self.ras(grad, n_ras) - offset
        return grad

    def fix_grid(self, grid):
        offset = -torch.min(grid)
        assert offset >= 0

        grid = grid + offset
        grid = grid / (grid.size(0) * offset + 1)

        assert torch.all(grid >= 0)
        return grid


def filter_tensor(tensor, revealed, only, fill=None):
    # If Only: only revealed will be returned
    # If Not Only: everything but revealed
    if not len(revealed):
        revealed = [[-1, -1]]

    revealed = torch.tensor(revealed, device=device)
    revealed_rows = revealed[:, 0]
    revealed_cols = revealed[:, 1]

    index_rows = torch.any(
        revealed_rows.unsqueeze(1)
        == torch.arange(
            (fill.size(0) if fill is not None else tensor.size(0)), device=device
        ).unsqueeze(0),
        dim=0,
    ).unsqueeze(1)
    index_cols = torch.any(
        revealed_cols.unsqueeze(1)
        == torch.arange(
            (fill.size(1) if fill is not None else tensor.size(1)), device=device
        ).unsqueeze(0),
        dim=0,
    ).unsqueeze(0)

    if not only:
        index_rows = ~index_rows
        index_cols = ~index_cols

    if fill is None:
        result = tensor[index_rows & index_cols]
        # square root
        size = int(len(result) ** 0.5)
        result = result.view(size, size)
        return result
    else:
        fill[index_rows & index_cols] = torch.flatten(tensor)
        return fill


import argparse

parser = argparse.ArgumentParser(description="Reversing Nearness via Gradient Descent")
parser.add_argument(
    "--n_optim", default=10, type=int, help="Number of optimization steps"
)
parser.add_argument(
    "--n_ras", default=5, type=int, help="Number of Sinkhorn-Knopp iterations"
)
parser.add_argument("--size", default=5, type=int, help="Size of the grid")

args = parser.parse_args()

writer = SummaryWriter(
    log_dir="runs/size" + str(args.size) + "optim" + str(args.n_optim)
)

if __name__ == "__main__":
    print("Size:", args.size, " n_optim:", args.n_optim)
    size = torch.tensor([args.size, args.size])
    # number of iterations
    grid = Grid(size)
    result, discrete_loss = grid.forward(args.n_optim, args.n_ras)
    writer.add_hparams(
        {"size": args.size, "n_optim": args.n_optim, "n_ras": args.n_ras},
        {"discrete_loss": discrete_loss},
    )

