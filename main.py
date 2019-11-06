import numpy as np
import torch
import torch.nn.functional as F
import toroid


def distanceCube(n):
    """Returns PyTorch distance cube
    Remember to only call once

    :n: Cube dimensions
    :returns: Cube

    """
    return torch.from_numpy(toroid.distanceCube(np.array([n, n])))


def superCube(n, rawGrid):
    """Performs softmax, then formats it into a cube

    :n: Cube dimensions
    :rawGrid: TODO
    :returns: TODO

    """
    rawGrid = F.softmax(rawGrid, dim=0)
    rawGrid = torch.unsqueeze(rawGrid, dim=0)
    rawGrid = rawGrid.expand(n**2, -1, -1)
    return rawGrid


def loss(n, rawGrid):
    """Returns loss

    :n: Cube dimensions
    :rawGrid: TODO
    :returns: TODO

    """
    value = superCube(n, rawGrid) * DISTANCE_CUBE
    value = torch.sum(value, dim=1)
    value = torch.triu(value)
    value = torch.sum(value)
    return value


if __name__ == "__main__":
    n = 4
    DISTANCE_CUBE = distanceCube(n)
