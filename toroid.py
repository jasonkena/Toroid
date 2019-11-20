import numpy as np


def two_to_one_coordinate(size, cell):
    return cell[0] * size[0] + cell[1]


def one_to_two_coordinate(size, coord):
    return np.array(np.divmod(coord, size[0]))


def toroid_distance2d(size, cellA, cellB):
    wrap_dist = np.absolute(size + np.minimum(cellA, cellB) - np.maximum(cellA, cellB))
    no_wrap_dist = np.absolute(cellA - cellB)
    min_dist = np.power(np.minimum(wrap_dist, no_wrap_dist), 2)
    return np.sum(min_dist)


def toroid_distance1d(size, coordA, coordB):
    return toroid_distance2d(
        size, *[one_to_two_coordinate(size, i) for i in [coordA, coordB]]
    )


def distance_grid(size):
    return np.array(
        [
            [toroid_distance1d(size, i, j) for i in range(np.prod(size))]
            for j in range(np.prod(size))
        ]
    )


def distance_cube(size):
    grid = distance_grid(size)
    grid = np.expand_dims(grid, axis=2)
    grid = np.repeat(grid, np.prod(size), axis=2)

    return grid
