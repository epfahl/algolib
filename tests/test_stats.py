"""Unit tests for stats functions.
"""

import numpy as np

from algolib import stats
from . import utils


def test_percentile_trim():
    ret = stats.percentile_trim(np.arange(10), percentile=20)
    res = (np.array([2, 3, 4, 5, 6, 7]), np.array([0, 1, 8, 9]))
    assert utils.nested_array_equal(res, ret)


def test_mode():
    assert stats.mode((1, 1, 2, 2, 2), weights=None) == 2
    assert stats.mode((1, 1, 2, 2, 2), weights=(10., 10., 1., 1., 1.)) == 1


def test_argpercentile():
    assert stats.argpercentile([1, 2, 3, 4, 5], 50) == 2
    assert stats.argpercentile([2, 1, 0, 1, 2, 3], 0) == 2
    assert stats.argpercentile([2, 1, 0, 1, 2, 3], 100) == 5


def test_argmin_last():
    assert stats.argmin_last([1, 1, 0, 0]) == 3
    assert stats.argmin_last([1, 1, 0, 0, 1, 1]) == 3
