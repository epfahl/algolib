"""Test segmentation functions.
"""

import numpy as np

from algolib import segments
from . import utils


def test_segment_slices():

    b = [1, 1, 0, 0, 1, 1, 1, 0]
    slices = segments.segment_slices(b)

    assert set([s.start for s in slices]) == {0, 4}
    assert set([s.stop for s in slices]) == {2, 7}


def test_segment_indices():

    b = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1])
    indices = segments.segment_indices(b)

    assert set(indices[:, 0]) == {0, 4, 8}
    assert set(indices[:, 1]) == {2, 6, 9}


def test_partitionby():
    a1 = [1, 1, -1, -1, 1, -1]
    a2 = [-1, -1, -1]
    a3 = [0, 1, 2, 4, 6, 5, 7, 8]
    p1 = segments.partitionby(lambda x: x == -1, a1)
    p2 = segments.partitionby(lambda x: x == -1, a2)
    p3 = segments.partitionby(lambda x: x % 2 == 0, a3)
    r1 = [np.array([1, 1]), np.array([-1, -1]), np.array([1]), np.array([-1])]
    r2 = [np.array([-1, -1, -1])]
    r3 = [
        np.array([0]), np.array([1]), np.array([2, 4, 6]),
        np.array([5, 7]), np.array([8])]
    assert utils.nested_array_equal(r1, p1)
    assert utils.nested_array_equal(r2, p2)
    assert utils.nested_array_equal(r3, p3)


def test_fpartitionby_indices():
    res = segments.fpartitionby_indices(lambda x: x == 0, [0, 0, 1, 1, 0])
    assert utils.nested_array_equal(res, [np.array([0, 1]), np.array([4])])


def test_fpartitionby_values():
    res = segments.fpartitionby_values(lambda x: x > 0, [1, 2, 0, 0, 3])
    assert utils.nested_array_equal(res, [np.array([1, 2]), np.array([3])])


def test_fpartitionby_fill():

    def _pfill(a):
        return segments.fpartitionby_fill(lambda x: x == -1, a, 2)

    a1 = [0, 1, -1, -1]
    a2 = [-1, -1, 2, 3]
    a3 = [0, -1, -1, -1]
    assert utils.array_equal(_pfill(a1), np.array([0, 1, 1, 1]))
    assert utils.array_equal(_pfill(a2), np.array([[2, 2, 2, 3]]))
    assert utils.array_equal(_pfill(a3), np.array([0, -1, -1, -1]))
