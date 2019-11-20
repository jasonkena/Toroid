"""
TODO
"""
import numpy as np
import torch
import torch.nn.functional as F
import toroid


def distance_cube(size):
    return torch.from_numpy(toroid.distanceCube(np.array([size, size])))


def super_cube(size, raw_grid):
    raw_grid = F.softmax(raw_grid, dim=0)
    raw_grid = torch.unsqueeze(raw_grid, dim=0)
    raw_grid = raw_grid.expand(size, -1, -1)
    return raw_grid


def loss(size, raw_grid):
    value = super_cube(size, raw_grid) * DISTANCE_CUBE
    value = torch.sum(value, dim=1)
    value = torch.triu(value)
    value = torch.sum(value)
    return value


if __name__ == "__main__":
    SIZE = 4
    DISTANCE_CUBE = distance_cube(SIZE)
