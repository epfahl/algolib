"""
Module for Outlier-Reslient Exponentially-Weighted Average (OREWA).
"""

# ------------------------------

import numpy as np

from . import bilateral


# ------------------------------

def _column_reshape(a):
    """
    Ensure that 1xN array-like objects are cast as Nx1 ndarrays.  Array-like
    objects with dimensions NxM(>=1) are converted to ndarrays, but their
    dimensions are unaffected.
    """

    return a.reshape(a.shape[0], -1)


def _ewa_weights(t, scale):
    """
    Given a vector of times and an exponential e-fold scale, return a vector
    of corresponding weights on the lookback times relative to the last point.
    """

    return np.exp(-((t[-1] - t)[::-1]) / float(scale))[::-1]


def _orewa_weights(t, y, tscale, zscale=None, positive_only=False):
    """
    Given aligned time and value vectors, as well scales for time and Z
    weighting, return a vector of net weights with unit sum.
    """

    return bilateral.bilateral_weights(
        y,
        init_weights=_ewa_weights(t, tscale),
        zscale=zscale,
        positive_only=positive_only
    )


# ------------------------------

def orewa1(
    time, data, tscale,
    nmax=None, target_column=0,
    zscale=None, positive_only=False
):
    """
    Return the Outlier-Reslient Exponentially-Weighted Average of the given
    time series data.  The last data point gets the largest temporal weight.

    Parameters
    ----------
    time: 1xN numerical array-like
        Ordered times
    data: NxM array-like
        Time series values
    tscale: float
        Time constant for exponential time w1eighting
    nmax: int
        Maximum number of points used for averaging (i.e., the lookback)
    target_column: int
        Index of the data column for which outliers are supporessed and sign
        is checked
    zscale: float
        e-fold scale for Gaussian suppression weight based on Z score
    positive_only: bool
        If True, values <= 0 in the target data column are given 0 weight

    Returns
    -------
    OREWA result: float
    """

    tary = np.atleast_1d(time)
    dary = _column_reshape(data)

    if nmax is None:
        nmax = tary.shape[0]

    avg = np.dot(
        _orewa_weights(
            tary[-nmax:], dary[-nmax:, target_column],
            tscale, zscale=zscale,
            positive_only=positive_only
        ),
        dary[-nmax:]
    )

    if avg.size == 1:
        return avg[0]
    else:
        return avg


def orewma(
    time, data, tscale,
    nmax=None, target_column=0,
    zscale=None, positive_only=False
):
    """
    Return the Outlier-Reslient Exponentially-Weighted Moving Average of the
    given time series data.

    Parameters
    ----------
    time: 1xN array-like
        Ordered times
    data: array-like
        Time series values
       tscale: float
        Time constant for exponential time w1eighting
    nmax: int
        Maximum number of points used for averaging (i.e., the lookback)
    target_column: int
        Index of the data column for which outliers are supporessed and sign
        is checked
    zscale: float
        e-fold scale for Gaussian suppression weight based on Z score
    positive_only: bool
        If True, values <= 0 in the target data column are given 0 weight

    Returns
    -------
    OREWMA result: ndarray with same shape as data
    """

    tary = np.atleast_1d(time)
    dary = np.atleast_1d(data)

    def _orewa(t, d):
        return orewa1(
            t, d, tscale, zscale=zscale,
            nmax=nmax, target_column=target_column, positive_only=positive_only
        )

    return np.array([
        _orewa(tary[:i + 1], dary[:i + 1])
        for i in range(tary.shape[0])
    ]).reshape(dary.shape)
