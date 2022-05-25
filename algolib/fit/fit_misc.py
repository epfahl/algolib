"""
Module containing utilities for fitting data.
"""

import functools

import numpy as np
from scipy import stats as spstats
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

from ..filters import bilateral


def _r2(fres, mres):
    """
    Return R2--the coefficient of determination under L2--given arrays of
    fit residuals and differences from the mean.
    """

    return 1. - (fres**2).sum() / (mres**2).sum()


def _r1(fres, mres):
    """
    Return R1--the coefficient of determination under L1--given arrays of
    measured and modeled values.
    """

    return 1. - np.abs(fres).sum() / np.abs(mres).sum()


def _std_weighted(y, weights=None):

    if weights is None:
        return y.std()

    return np.sqrt(np.average(
        (y - np.average(y, weights=weights))**2,
        weights=weights
    ))


def _knn_indices1d(x0, x, k):
    """
    Return an array of indices of the k nearest neighbors of x0 in the 1D
    array x.

    * This is probably dumb.  If the vector is already sorted, can march in
      indices, rather than values.
    """

    return np.abs(x0 - x).argsort()[:k]


def _data_knn1d(xp, data, k):
    """
    Return the kNN data segments for each choice of predictor variable.
    """

    return data[np.array([_knn_indices1d(x, data[:, 0], k) for x in xp])]


def _kernel_weights(x, kernel_scale, kernel_func):
    """
    Return an array of kernel weights, given a vector of x values, kernel
    scale, and the name of a function in scipy.stats.
    """

    weights = getattr(spstats, kernel_func)(
        loc=0., scale=kernel_scale
    ).pdf(x)

    return weights / weights.sum()


def _line_fit1d(data, weights=None):
    """
    Return the poly1d object representing the weighted linear regression,
    given arrays of data (Nx2) and weights (Nx1).
    """

    return np.poly1d(np.polyfit(data[:, 0], data[:, 1], 1, w=weights))


def _linear_predict1d(
    x, data, kernel_scale=None, kernel_func=None, init_weights=None
):

    if init_weights is None:
        iweights = np.ones(data.shape[0])
    else:
        iweights = np.atleast_1d(init_weights)

    if kernel_scale is None:
        kweights = None
    else:
        if kernel_func is None:
            kernel_func = 'norm'
        kweights = _kernel_weights(data[:, 0] - x, kernel_scale, kernel_func)

    # Net weights
    weights = iweights * kweights

    poly1d = _line_fit1d(data, weights)

    return (
        poly1d(x),
        _std_weighted(data[:, 1] - poly1d(data[:, 0]), weights=weights)
    )


def fitness(y, yfit):
    """
    Given the data and fit values, return a dict of fitness metrics.
    """

    ya = np.array(y)
    yfa = np.array(yfit)

    fres = ya - yfa
    mres = ya - ya.mean()

    return {
        'r1': _r2(fres, mres),
        'r2': _r1(fres, mres),
    }


def polyfit(x, y, degree, weights=None):
    """
    Fit a polynomial to the given and return a dict of commonly used info.

    Parameters
    ----------
    x: array-like
        Independent variable
    y: array-like
        Dependent variable; data to be modeled
    degree: int
        Polynomial degree
    weights: array-like
        Vector of weights applied to sample values

    Returns
    -------
    info: dict
        {
            'coeff': (numpy.ndarray) polynovial coefficients,
            'cov': (numpy.ndarray) covariance matrix,
            'poly1d': (numpy.poly1d object) complete fit characterization,
            'residual': (numpy.ndaray) fit residuals,
            'std_residual': (float) weighted std of the residuals,
            'r2': (float) Global L2 coefficient of determination,
        }

    Examples
    --------
    >>> x = np.arange(1, 6)
    >>> y = x
    >>> get_polyfit(x, y)
    {'poly1d': poly1d([  1.00000000e+00,   1.98602732e-16]),
     'r1': 0.99999999999999956,
     'r2': 1.0,
     'std_residual': 3.8714799753065006e-16}

    ToDo
    ----
    * Build this out to allow outlier suppression.
    """

    coeff, cov = np.polyfit(x, y, degree, w=weights, cov=True)
    poly1d = np.poly1d(coeff)
    fit = poly1d(x)
    fit_res = y - fit
    mean_res = y - y.mean()

    return {
        'fit': fit,
        'coeff': coeff,
        'cov': cov,
        'poly1d': poly1d,
        'residual': fit_res,
        'std_residual': _std_weighted(fit_res, weights=weights),
        'r2': _r2(fit_res, mean_res),
    }


def knn_mean_regression1d(xp, data, k):
    """
    General predictions and local variances from the provided x-y data using
    kNN regression.  Each prediction is the mean of the kNN sample.

    Parameters
    ----------
    xp: 1xN array-like
    data: Nx2 numpy.ndarray
    k: int

    Returns
    -------
    Nx2 numpy.ndarray

    Examples
    --------
    """

    data_knn = _data_knn1d(np.atleast_1d(xp), data, k)

    return np.array([
        data_knn[:, :, 1].mean(axis=1),
        data_knn[:, :, 1].std(axis=1)
    ]).T


def knn_linear_regression1d(xp, data, k):
    """
    Generate predictions and local variances from the provided x-y data using
    kNN regression.  Each prediction obtained from a linear fit to the
    kNN sample.

    Parameters
    ----------
    xp: 1xN array-like
    data: Nx2 numpy.ndarray
    k: int
    max_extrap: float

    Returns
    -------
    Nx2 numpy.ndarray

    Examples
    --------
    """

    return np.array([
        _linear_predict1d(x, _data_knn1d(x, data, k)[0])
        for x in np.atleast_1d(xp)
    ])


def kernel_linear_regression1d(
    xp, data, kernel_scale, kernel_func='norm', init_weights=None,
    suppress_outliers=False, outlier_zscale=None, positive_only=False
):
    """
    Generate predictions and local variances from the provided x-y data using
    kernel-localized regression.  Each prediction is obtained from a linear
    fit to kernel-weighted data, where kernel falls off as the proximity from
    the predictor value increases.

    Parameters
    ----------
    xp: 1xN array-like
    data: Nx2 numpy.ndarray
    kernel_scale: float
    kernel_func: str ('norm')
    init_weights: 1xN array-like (None)
    suppress_outliers: bool (False)
    outlier_zscale: float (None)
    positive_only: bool

    Returns
    -------
    Nx2 numpy.ndarray

    Examples
    --------
    """

    predict = functools.partial(
        _linear_predict1d,
        kernel_scale=kernel_scale, kernel_func=kernel_func
    )

    if suppress_outliers and (outlier_zscale is not None):
        pred0 = np.array([predict(x, data, init_weights=None)
                          for x in data[:, 0]])
        init_weights = bilateral._gaussian_zweights(
            bilateral._zscores(data[:, 1] - pred0[:, 0],
                               positive_only=positive_only),
            outlier_zscale
        )

    return np.array([
        predict(x, data, init_weights=init_weights)
        for x in np.atleast_1d(xp)
    ])


def _frac_from_scale(x, scale):

    counts, _ = np.histogram(
        x, bins=int((x.max() - x.min()) / float(scale))
    )

    return max(5., 0.5 * (counts.min() + counts.max())) / x.size


def _scale_from_frac(x, frac):

    return frac * np.subtract(*np.percentile(x, (90, 10)))


def _delta_from_scale(scale):

    return 0.25 * scale


def lowess(x, y, frac=None, knn=None, scale=None, is_sorted=False):
    """
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not is_sorted:
        argsort = x.argsort()
        x = x[argsort]
        y = y[argsort]

    if all([kw is None for kw in (frac, knn, scale)]):
        frac = 2. / 3.  # statsmodels default
        knn = frac * x.size
        scale = _scale_from_frac(x, frac)
    elif (frac is not None) and (knn is None) and (scale is None):
        knn = frac * x.size
        scale = _scale_from_frac(x, frac)
    elif (frac is None) and (knn is not None) and (scale is None):
        frac = float(knn) / x.size
        scale = _scale_from_frac(x, frac)
    elif (frac is None) and (knn is None) and (scale is not None):
        frac = _frac_from_scale(x, scale)
        knn = int(frac * x.size)
    else:
        raise ValueError("only one of (frac, knn, scale) can be non-null")

    return _lowess(
        y, x, frac=frac, it=2, delta=_delta_from_scale(scale), is_sorted=True
    )
