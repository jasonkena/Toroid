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
        self.new_size = (size ** 2).tolist()

        self.scaling_threshold = scaling_threshold
        self.grid = torch.rand(self.new_size, requires_grad=True, device=device)
        self.scale_a, self.scale_b = [
            torch.ones(torch.prod(size).tolist(), requires_grad=True, device=device)
            for _ in range(2)
        ]
        self.scale_optim = optim.Adam([self.scale_a, self.scale_b], lr=1e-3)
        self.grid_optim = optim.SGD([self.grid], lr=3e-4)

        eq = "ac,bd,cd,ab,ab->"
        self.distance_grid = calculate_distance_grid(size).to(device)
        self.triu = torch.triu(torch.ones((size ** 2).tolist())).to(device)

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
        try:
            self.ras()
            new_grid = self.scale_a @ self.grid @ self.scale_b
            grid_loss = self.expr(new_grid, new_grid, backend="torch")
            grid_loss.backward()
            self.grid_optim.step()
        except KeyboardInterrupt:
            pass
        print("Grid Loss:", grid_loss)
        return grid_loss

    def scaling_loss(self, scale_a, scale_b):
        resultant = scale_a @ self.grid @ scale_b
        row_sum = torch.sum(resultant, axis=0)
        row_column = torch.sum(resultant, axis=1)

        # This is not mean loss
        loss = torch.sum((row_sum - 1) ** 2 + (row_column - 1) ** 2)
        return loss

    def ras(self, check_frequency=10):
        grid = self.grid.detach()
        scale_a, scale_b = [
            torch.diag(torch.ones(self.new_size[0], device=device)) for _ in range(2)
        ]

        counter = 0

        while True:
            if counter % check_frequency == 0:
                scaling_loss = self.scaling_loss(scale_a, scale_b)
                print("Scaling loss", scaling_loss)
                if scaling_loss <= self.scaling_threshold:
                    break
            newscale_a = torch.diag(1 / torch.sum(grid, axis=1))
            assert not torch.isnan(newscale_a).any()
            grid = newscale_a @ grid
            assert not torch.isnan(grid).any()
            print(scale_a)
            scale_a = newscale_a @ scale_a
            assert not torch.isnan(scale_a).any()

            newscale_b = torch.diag(1 / torch.sum(grid, axis=0))
            grid = grid @ newscale_b
            scale_b = scale_b @ newscale_b
            counter = counter + 1

        self.scale_a, self.scale_b = scale_a, scale_b
        return scale_a, scale_b


if __name__ == "__main__":
    size = torch.tensor([10, 10])
    grid = Grid(size)

