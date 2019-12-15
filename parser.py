import string
import toroid
import numpy as np


def dictionary():
    return string.ascii_uppercase + "1234"


def unparse(string, dictionary):
    array = [dictionary.index(i) for i in string if i in dictionary]
    array = np.array([array[i : i + 2] for i in range(0, len(array), 2)])
    size = np.array([np.sqrt(array.shape[0])] * 2)
    array = np.array(
        [toroid.two_to_one_coordinate(size, i) for i in array], dtype=np.int32
    )
    result = np.zeros([np.prod(size).astype(int)] * 2)
    for i in np.arange(np.prod(size).astype(int)):
        result[i][array[i]] = 1
    return result
