"""
size is Size of Original grid, from 6 to 30
"""
import numpy as np
import torch
import torch.nn.functional as F
import toroid
import parser

def softmax(raw_grid):
    return

def distance_grid(size):
    return torch.from_numpy(toroid.distance_grid(size.numpy())).type(torch.FloatTensor)


def read(filename):
    size, grid = parser.read(filename)
    return torch.from_numpy(size), torch.from_numpy(grid).type(torch.FloatTensor)


def default(size):
    return torch.eye(torch.prod(size))


def loss(raw_grid, distance_grid):
    result = torch.einsum(
        "ab,ac,bd,cd->ab", distance_grid, raw_grid, raw_grid, distance_grid
    )
    return torch.triu(result)


if __name__ == "__main__":
    size, grid = read("sample3")
    a = loss(grid, distance_grid(size))

