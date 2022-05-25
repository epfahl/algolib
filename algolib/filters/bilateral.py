"""
Module for bilateral filtering and related ideas.

* Right now, only bilateral averaging is implemented.
* Should I have _zscores_standard and _zscores_percentile?
"""

import numpy as np

from . import misc


# ------------------------------

def _zscores(y, plow=25, phigh=75, positive_only=False):
    """
    Given a data vector, return percentile-based Z magnitudes.  Optionally
    given negative values 0 weight.
    """

    if y.shape[0] <= 1:
        return 0.

    if positive_only:
        ypos = y[y > 0]
        if ypos.shape[0] <= 1:
            return 0.
        median = np.median(ypos)
        variance = misc.ipr(ypos, plow, phigh)
    else:
        median = np.median(y)
        variance = misc.ipr(y, plow, phigh)

    if variance == 0.:
        return np.abs(y - median)
    else:
        return np.abs(y - median) / variance


def _gaussian_zweights(z, scale):
    """
    Given a vector of Z scores an a Gaussian e-fold scale, return a
    vector of corresponding weights.
    """

    return np.exp(-z**2 / scale**2)


def _negative_weights(y):
    """
    Return a binary weighting vector that is 0 when y <= 0 and 1 otherwise.
    """

    return (y > 0.).astype(float)


# ------------------------------

def bilateral_weights(y, init_weights=None, zscale=None, positive_only=False):
    """
    Given a vector of values, a vector of initial weights (defaulted to 1),
    and a scale for Z weights, return a vector of net bilateral weights with
    unit sum.
    """

    y = np.atleast_1d(y)

    # Set initial weights to 1 if none provided
    if init_weights is None:
        init_weights = np.ones(y.shape[0])
    else:
        init_weights = np.atleast_1d(init_weights)
    w = init_weights

    # Weight NaN values by 0
    w *= (~np.isnan(y)).astype(float)

    # Z-based weights, if scale given
    if zscale is not None:
        w *= _gaussian_zweights(
            _zscores(y, positive_only=positive_only),
            zscale
        )

    # Weight negative data by 0 if requested
    if positive_only:
        w *= _negative_weights(y)

    # Return sum-normed weights
    return np.nan_to_num(w / w.sum())


def bilateral_average(y, init_weights=None, zscale=None, positive_only=False):
    """
    Return the bilateral average of the given 1D data.

    Parameters
    ----------
    y: 1xN array-like
        Input 1D data
    init_weights: 1xN array-like
        Initial weights applied to data; defaults to np.ones(len(y)) if not
        given
    zscale: float
        If not None, the Z scale above which a point gets strong Gaussian
        suppressed; if None, Z weighting is not applied
    positive_only: bool
        If True, negative data is given weight 0

    Returns
    -------
    bilateral average: float

    Examples
    --------
    >>> y = np.ones(10)
    >>> y[5] = 10
    >>> bilateral_average(y)
    1.9000000000000004
    >>> np.mean(y)
    1.8999999999999999
    >>> bilateral_average(y, zscale=3)
    1.0001234081118899

    Notes
    -----
    When init_weights == None, zscale == None, and positive_only=False, this
    function just returns the ordinary mean.
    """

    yary = np.atleast_1d(y)

    return np.dot(
        bilateral_weights(
            yary,
            init_weights=init_weights,
            zscale=zscale,
            positive_only=positive_only
        ),
        yary
    )
