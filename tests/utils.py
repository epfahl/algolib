import numpy as np


def array_equal(a, b):
    return np.all(a == b)


def nested_array_equal(a, b):
    return all([array_equal(x, y) for x, y in zip(a, b)])
