"""Utilities for segmenting data.
"""

import numpy as np
import scipy.ndimage as ndi


def segment_slices(binary):
    """
    Find the slices of segments for which the given binary/boolean vector is
    1/True.

    Parameters:
    binary: array-like, 1D
        Input binary/boolean vector

    Returns
    -------
    slices: list
        List of slice objects

    Examples
    --------
    >>> binary = [1, 1, 0, 0, 1, 1, 1, 0]
    >>> segment_slices(binary)
    [slice(0L, 2L, None), slice(4L, 7L, None)]
    """

    slices = ndi.find_objects(ndi.label(binary)[0])

    if slices != [()]:
        slices = [s[0] for s in slices]
    else:
        slices = []

    return slices


def segment_indices(binary):
    """
    Find the start/stop indices of all contiguous segments in a binary/boolean
    vector for which the value is 1 or True.

    Parameters
    ----------
    binary: binary/boolean sequence

    Returns
    -------
    segments: (Nx2) array of ints
        (i_start, i_stop) index pairs, where i_stop is 1 more than the last
        True element in the segment

    Examples
    --------
    >>> b = [1, 1, 0, 0, 1, 1, 0, 0, 1]
    >>> segment_indices(b)
    array([[0, 2],
           [4, 6],
           [8, 9]])
    """
    binary_diff = np.ediff1d(np.concatenate(([0], binary.astype(int), [0])))
    return np.array((
        (binary_diff == 1).nonzero()[0],
        (binary_diff == -1).nonzero()[0])).T


def partitionby(boolfn, ary):
    """Return a list of contiguous subarrays of the given array, where each
    subarray is a run of True or False according to the boolean function.

    Examples
    --------
    >>> partitionby(lambda x: x == -1, [1, 1, -1, -1, 1, -1])
    [array([1, 1]), array([-1, -1]), array([1]), array([-1])]
    >>> partitionby(lambda x: x == -1, [-1, -1, -1])
    [array([-1, -1, -1])]
    >>> partitionby(lambda x: x%2 == 0, [0, 1, 2, 4, 6, 5, 7, 8])
    [array([0]), array([1]), array([2, 4, 6]), array([5, 7]), array([8])]
    """
    return np.split(
        ary,
        np.nonzero(
            np.ediff1d(
                boolfn(np.array(ary)).astype(int)) != 0)[0] + 1)


def fpartitionby_indices(fn, seq):
    """Filtered partitionby that returns a list of arrays of sequence
    _indices_ corresponding to partitions for which the function, when applied
    to the sequence, evaluates to 'truthy'.

    Examples
    --------
    >>> fpartitionby_indices(lambda x: x == 0, [0, 0, 1, 1, 0])
    [array([0, 1]), array([4])]
    """
    b = fn(np.array(seq)).astype(bool)
    idx = np.arange(b.size)[b]
    ret = np.split(idx, np.nonzero(np.ediff1d(idx) != 1)[0] + 1)
    if (len(ret) == 1) and (len(ret[0]) == 0):
        return []
    else:
        return ret


def fpartitionby_values(fn, seq):
    """Filtered partitionby that returns a list of arrays of sequence
    _values_ corresponding to partitions for which the function, when applied
    to the sequence, evaluates to 'truthy'.

    Examples
    --------
    >>> fpartitionby_values(lambda x: x > 0, [1, 2, 0, 0, 3])
    [array([1, 2]), array([3])]
    """
    return map(lambda p: np.array(seq)[p], fpartitionby_indices(fn, seq))


def fpartitionby_fill(fn, seq, nmax):
    """For segments of length <= nmax, where the function, when applied to the
    sequence, returns 'truthy', fill with neighboring values.  Filling from the
    nearest 'falsy' value on the left is preferred over filling from the right.

    Examples
    --------
    >>> fpartitionby_fill(lambda x: x == -1, [0, 1, -1, -1], 2)
    array([0, 1, 1, 1])
    >>> fpartitionby_fill(lambda x: x == -1, [-1, -1, 2, 3], 2)
    array([[2, 2, 2, 3]])
    >>> fpartitionby_fill(lambda x: x == -1, [0, -1, -1, -1], 2)
    array([0, -1, -1, -1])
    """
    ary = np.array(seq)
    parts = filter(lambda p: p.size <= nmax, fpartitionby_indices(fn, seq))
    if (len(parts) == 0) or (len(parts[0]) == ary.size):
        return ary

    def _fill(p):
        ifirst, ilast = p[0], p[-1]
        if ifirst == 0:
            ary[p] = ary[ilast + 1]
        else:
            ary[p] = ary[ifirst - 1]

    map(_fill, parts)
    return ary
