"""Statistical utilities for 1D data.
"""

import numpy as np


def percentile_trim(data, percentile=None):
    """
    Trim the upper and lower percentiles from the given 1D data along the given
    axis.  Returns both the trimmed data and  the indices of points that were
    cut.

    Parameters
    ----------
    data: numpy.array
        Input 1D array
    percentile: float/int
        Points less than the lower percentile and higher than the upper
        percentile will be removed
    axis: int
        Axis along which the trimming is done

    Returns
    -------
    data_trimmed: numpy.ndarray
        Subset of original data with
    idx_cut: numpy.ndarray
        Indices in the original data of points that were cut

    Examples
    --------
    >>> data = np.arange(10)
    >>> percentile_trim(data, percentile=20)
    (array([2, 3, 4, 5, 6, 7]), array([0, 1, 8, 9]))
    """

    if percentile is None or percentile == 0:
        return data, np.array([])

    lower = np.percentile(data, q=percentile)
    upper = np.percentile(data, q=100. - percentile)
    trim_bool = (data >= lower) & (data <= upper)

    data_trimmed = data[trim_bool]
    idx_cut = (~trim_bool).nonzero()[0]

    return data_trimmed, idx_cut


def argpercentile(a, p):
    """
    Return the array index of the value closest to the given percentile.

    Parameters
    ----------
    a: array-like (1D)
        Input array
    p: float/int
        Percentile in interval [0, 100]

    Examples
    --------
    >>> argpercentile([1, 2, 3, 4, 5], 50)
    2
    >>> argpercentile([2, 1, 0, 1, 2, 3], 0)
    2
    >>> argpercentile([2, 1, 0, 1, 2, 3], 100)
    5
    """
    if p == 0:
        return np.argmin(a)
    elif p == 100:
        return np.argmax(a)
    else:
        return np.abs(a - np.percentile(a, p)).argmin()


def ipr(a, p1, p2, axis=None):
    """Return the interpercentile range of the given array.
    """
    return np.subtract(*np.percentile(a, sorted([p1, p2])[::-1], axis=axis))


def iqr(a, axis=None):
    """Return the interquartile range of the given array.
    """
    return ipr(a, 25, 75, axis=axis)


def rms(a):
    """
    Return the root-mean-square of the given 1D numerical sequence.
    """

    return np.sqrt(np.mean(np.atleast_1d(a)**2))


def mode(values, weights=None):
    """Return the (weighted) mode of the given hashable values.  If weights
    are given, the mode is the value for which the sum of weights is
    maximized.

    Parameters
    ----------
    values: 1D array-like
    weights[None]: 1D array-like (same length as values)

    Examples
    --------
    >>> mode((1, 1, 2, 2, 2), weights=None)
    2
    >>> mode((1, 1, 2, 2, 2), weights=(10., 10., 1., 1., 1.))
    1
    """
    values = np.atleast_1d(values)
    if weights is None:
        weights = np.ones(values.size)
    else:
        weights = np.atleast_1d(weights).astype(float)
    if values.size != weights.size:
        raise TypeError("len(weights) != len(values)")
    uvalues = np.unique(values)
    return uvalues[
        ((values == np.vstack(uvalues)) * weights).sum(axis=1).argmax()]


def argmin_last(a):
    """Return the *last* index corresponding to the minimum value of the given
    1D array.  By contrast, argmin returns the *first* index for which the
    value is the minimum.

    Parameters
    ----------
    a: 1D array-like

    Examples
    --------
    >>> argmin_last([1, 1, 0, 0])
    3
    >>> argmin_last([1, 1, 0, 0, 1, 1])
    3
    """
    return len(a) - 1 - np.argmin(a[::-1])
