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
    def __init__(self, size, scaling_threshold=1):
        # Size is a torch tensor, BEFORE HyperGrid
        super(Grid, self).__init__()
        # import pdb; pdb.set_trace()
        self.scaling_threshold = scaling_threshold
        self.grid = torch.rand((size ** 2).tolist(), requires_grad=True, device=device)
        self.scale_a, self.scale_b = [
            torch.ones(torch.prod(size).tolist(), requires_grad=True, device=device)
            for _ in range(2)
        ]
        self.scale_optim = optim.Adam([self.scale_a, self.scale_b], lr=1e-3)
        self.grid_optim = optim.Adam([self.grid], lr=1e-5)

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
            raise MyException("Broken")

        # Calculate Scaling
        # nn.init.ones_(self.scale_a)
        # nn.init.ones_(self.scale_b)
        # scaling_loss = self.scaling_loss()
        # while scaling_loss > self.scaling_threshold:
        # self.scale_optim.zero_grad()
        # scaling_loss = self.scaling_loss()
        # scaling_loss.backward()
        # self.scale_optim.step()
        # print("Scaling Loss:", scaling_loss)
        # if torch.isnan(self.scale_a).any() or torch.isnan(self.scale_b).any():
        # raise MyException("Nan")
        # new_grid = torch.diag(self.scale_a) @ self.grid @ torch.diag(self.scale_b)
        new_grid = self.grid
        scale_loss = self.implicit_scaling_loss()
        grid_loss = self.expr(new_grid, new_grid, backend="torch")

        total_loss = scale_loss + grid_loss
        total_loss.backward()

        self.grid_optim.step()
        print("Grid Loss:", grid_loss)
        # return [scaling_loss, grid_loss]
        # return grid_loss

    def scaling_loss(self):
        grid = self.grid.detach()
        resultant = torch.diag(self.scale_a) @ grid @ torch.diag(self.scale_b)
        row_sum = torch.sum(resultant, axis=0)
        row_column = torch.sum(resultant, axis=1)

        # This is not mean loss
        loss = torch.sum((row_sum - 1) ** 2 + (row_column - 1) ** 2)
        return loss

    def implicit_scaling_loss(self):
        row_sum = torch.sum(self.grid, axis=0)
        row_column = torch.sum(self.grid, axis=1)

        # This is not mean loss
        loss = torch.sum((row_sum - 1) ** 2 + (row_column - 1) ** 2)
        return loss

    # def ras(self, max_iter=10, iteration=0):
    # torch.div(self.grid, torch.sum(a, axis=0), out=self.grid)
    # a = a / (np.sum(a, axis=1)[:, np.newaxis])
    # # print(a)
    # if iteration >= max_iter:
    # print(a)
    # return a
    # return f(a, max_iter, iteration + 1)

