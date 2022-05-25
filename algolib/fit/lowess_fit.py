"""LOWESS implementation using the scikit-learn fit-predict framework.

Notes
-----
* In the future, choose scale as the only way to select the neighborhood size.
"""

import numpy as np
from statsmodels.nonparametric import smoothers_lowess

from ..stats import stats


NEXTRAP = 4


class LowessNotEnoughFitData(ValueError):
    pass


class LowessMismatchedInputSizes(ValueError):
    pass


# Data Wrangling

def _flat1d(x):
    """Return a flat ndarray of floats.
    """
    return np.ravel(np.asarray(x, dtype=float))


def _any_nan(x):
    """Return True if there are any nan values.
    """
    return np.any(np.isnan(x))


# Adaptive Parameter Parsing

def _is_none(*args):
    """List of bools; True if None."""

    return [v is None for v in args]


def _frac_from_scale(x, scale):
    """Frac from scale using a set of heuristics."""

    return min(1., scale / stats.ipr(x, 10, 90))


def _scale_from_frac(x, frac):
    """Scale from frac using a percentile range."""

    return frac * np.subtract(*np.percentile(x, (90, 10)))


def _delta_from_scale(scale):
    """Delta as a fraction of the scale."""

    return 0.2 * scale


def _parse_pars(x, frac, k, scale, delta):
    """Return a complete set of parameters from (possibly null) inputs."""

    is_none = _is_none(frac, k, scale)
    size = x.size

    if is_none == [False, True, True]:
        k = int(frac * size)
        scale = _scale_from_frac(x, frac)
    elif is_none == [True, False, True]:
        frac = float(k) / size
        scale = _scale_from_frac(x, frac)
    elif is_none == [True, True, False]:
        frac = _frac_from_scale(x, scale)
        k = int(frac * size)
    else:
        frac = 2. / 3.  # statsmodels default
        k = int(frac * size)
        scale = _scale_from_frac(x, frac)

    if delta is None:
        delta = 0.

    return frac, k, scale, delta


class LOWESS(object):
    """
    Compute a linear LOWESS model of 2D data.

    Parameters
    ----------
    frac [None]: float
        Fraction of data used to regress around each point
    k [None]: int
        Number of nearest neighbors used to regress around each point
    scale [None]: float
        Kernel scale for regression around each point
    delta [None]: float
        Neighborhood within which points ignored for regression

    Methods
    -------
    fit(x,y,[is_sorted]): Compute regression on raw data
    predict(x,[is_sorted]): Predict at given points, using linear extrapolation
        outside range of raw data

    Attributes
    ----------
    regression: Nx2 numpy.ndarray
        Regression of the raw data after running LOWESS.fit
    """

    def __init__(self, frac=None, k=None, scale=None, delta=None, it=2):

        if sum(1 for x in (frac, k, scale) if x is not None) > 1:
            raise ValueError("Only one of (frac, k, scale) can be non-null.")

        self.frac = frac
        self.k = k
        self.scale = scale
        self.delta = delta
        self.it = it

    def fit(self, x, y, is_sorted=False):
        """Obtain the LOWESS regression on the given data.

        Parameters
        ----------
        x: array-like
            Predictor data
        y: array-like
            Response data
        is_sorted [False]: bool
            If True, assumes input data is sorted by x

        Returns
        -------
        Instance of LOWESS full parameter set and regression data as
        attributes.
        """

        x = _flat1d(x)
        y = _flat1d(y)

        if _any_nan(x):
            raise ValueError("the x array contains one or more NaNs")

        if _any_nan(y):
            raise ValueError("the y array contains one or more NaNs")

        if len(x) != len(y):
            raise LowessMismatchedInputSizes()

        if len(x) < 2 or len(y) < 2:
            raise LowessNotEnoughFitData()

        self.xmin, self.xmax = x.min(), x.max()

        (
            self.frac, self.k, self.scale, self.delta
        ) = _parse_pars(x, self.frac, self.k, self.scale, self.delta)

        self.regression = np.array([
            x,
            smoothers_lowess.lowess(
                y, x, frac=self.frac, delta=self.delta, it=self.it,
                is_sorted=is_sorted, return_sorted=False
            )
        ]).T
        self.residual = y - self.regression[:, 1]

        return self
