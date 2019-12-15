"""
size is Size of Original grid, from 6 to 30
"""
import numpy as np
import torch
import torch.nn.functional as F
import toroid
import parser


def distance_cube(size):
    return torch.from_numpy(toroid.distance_cube(size.numpy())).type(torch.FloatTensor)


def super_cube(size, raw_grid):
    # raw_grid = F.softmax(raw_grid, dim=0)
    raw_grid = torch.unsqueeze(raw_grid, dim=0)
    raw_grid = raw_grid.expand(torch.prod(size), -1, -1)
    return raw_grid


def default(size):
    return torch.eye(torch.prod(size))


def loss(size, raw_grid, distance_cube):
    value = super_cube(size, raw_grid) * distance_cube
    value = torch.sum(value, dim=1)
    value = torch.triu(value)
    # COMMENTED
    # value = torch.sum(value)
    return value


# PLEASE REFORMAT
def hyperloss(size, raw_grid, distance_cube):
    return torch.sum(
        loss(size, raw_grid, distance_cube)
        * loss(size, torch.eye(torch.prod(size)), distance_cube)
    )


if __name__ == "__main__":

    print(
        hyperloss(
            torch.tensor([5, 5]),
            torch.from_numpy(
                parser.unparse(open("sample").read(), parser.dictionary())
            ).type(torch.FloatTensor),
            distance_cube(torch.tensor([5, 5])),
        )
    )
