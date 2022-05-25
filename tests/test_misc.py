"""Unit tests for uncategorized utiltiies.
"""

import numpy as np

from algolib import misc
from . import utils


def test_sloppy_intersect():
    a1 = np.array([1, 10, 20])
    a2 = np.array([3, 10, 21, 17])
    m1, m2 = misc.sloppy_intersect(a1, a2, slop=1)
    assert utils.array_equal(a1[m1], np.array([10, 20]))
    assert utils.array_equal(a2[m2], np.array([10, 21]))


def test_bimodal_partition():
    np.random.seed(1234)
    g1 = np.random.randn(100) - 10.
    np.random.seed(1235)
    g2 = np.random.randn(100) + 10.
    a = np.concatenate((g1, g2))
    assert misc.bimodal_partition(np.array([-10.0, 10.0])) == 0.0
    assert round(misc.bimodal_partition(a), 3) == -0.409


def test_lazy_join():
    assert misc.lazy_join([{1, 2}, {1, 3}, {5, 6}]) == [{1, 2, 3}, {5, 6}]


def test_pad_ends():
    a = np.arange(5)
    assert utils.array_equal(
        misc.pad_ends(a, mode='copy'),
        np.array([0, 0, 1, 2, 3, 4, 4]))
    assert utils.array_equal(
        misc.pad_ends(a, mode='constant', constant=-1),
        np.array([-1, 0, 1, 2, 3, 4, -1]))
