import string
import toroid
import numpy as np


def dictionary():
    return string.ascii_uppercase + "1234"


def unparse(string, dictionary):
    # String should be open("sample")

    # Return index per character of string
    array = [dictionary.index(i) for i in string if i in dictionary]

    # Group 2 consecutive coordinates
    array = np.array([array[i : i + 2] for i in range(0, len(array), 2)])

    # Array for the size, size is not Superpositioned yet
    size = np.array([np.sqrt(array.shape[0])] * 2)

    # Parse 2d coordinates into 1d coordinates, in a single list
    array = np.array(
        [toroid.two_to_one_coordinate(size, i) for i in array], dtype=np.int32
    )

    # Setup a Superposition grid
    result = np.zeros([np.prod(size).astype(int)] * 2)

    for i in np.arange(np.prod(size).astype(int)):
        result[i][array[i]] = 1
    return size, result


def read(filename):
    with open(filename, "r") as file:
        raw = file.read()
    return unparse(raw, dictionary())
