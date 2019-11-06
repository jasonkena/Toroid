import numpy as np
import torch


def twoToOneCoordinate(n, cell):
    """Returns converted coordinates, from 2 dimensions to 1 dimension

    :n: Size of toroidal grid with x and y components
    :cell: Array with x and y componenets
    :returns: Single integer representing coordinate

    """
    return cell[0] * n[0] + cell[1]


def oneToTwoCoordinate(n, coord):
    """Returns converted coordinates, from 1 dimension to 2 dimensions

    :n: Size of toroidal grid with x and y components
    :coord: Single integer representing coordinates
    :returns: Array representing coordinates

    """
    return np.array(np.divmod(coord, n[0]))


def toroidDistance2d(n, cellA, cellB):
    """Returns squared Euclidean distance between 2 coordinates on a toroidal grid

    :n: Size of toroidal grid with x and y components
    :cellA: Array with x and y components
    :cellB: Array with x and y components
    :returns: Squared integer distance between 2 coordinates
    """
    wrapDist = np.absolute(n + np.minimum(cellA, cellB) -
                           np.maximum(cellA, cellB))
    noWrapDist = np.absolute(cellA - cellB)
    minDist = np.power(np.minimum(wrapDist, noWrapDist), 2)
    return np.sum(minDist)


def toroidDistance1d(n, coordA, coordB):
    """Wrapper around toroidDistance2d, to use 1d coordinates

    :n: Size of toroidal grid with x and y components
    :coordA: Coordinate A
    :coordB: Coordinate B
    :returns: Squared integer distance between 2 coordinates

    """
    return toroidDistance2d(
        n, *[oneToTwoCoordinate(n, i) for i in [coordA, coordB]])


def distanceGrid(n):
    """Returns matrix of distances between any point

    :n: Size of toroidal grid with x and y components
    :returns: Matrix of distances

    """
    return np.array([[toroidDistance1d(n, i, j) for i in range(np.prod(n))]
                     for j in range(np.prod(n))])


def distanceCube(n):
    """TODO

    :n: Size of toroidal grid with x and y components
    :returns: TODO

    """
    grid = distanceGrid(n)
    grid = np.expand_dims(grid, axis=2)
    grid = np.repeat(grid, np.prod(n), axis=2)

    return grid
