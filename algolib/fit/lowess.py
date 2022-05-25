"""
LOWESS implementation using the scikit-learn fit-predict framework.

Notes
-----
* This is a dump of the original implementation that includes both the fit
  and predict methods.
"""

import numpy as np
from scipy import interpolate
from statsmodels.nonparametric import smoothers_lowess

from ..stats import stats
from ..misc import misc

NEXTRAP = 4


# ------------------------------

# Data Wrangling

def _flat1d(x):
    """Return a flat ndarray of floats.
    """
    return np.ravel(np.asarray(x, dtype=float))


def _any_nan(x):
    """Return True if there are any nan values.
    """
    return np.any(np.isnan(x))


def _sortby(ary, col):
    """Sort 2D array by the given column index.
    """
    return ary[ary[:, col].argsort()]


# ------------------------------

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


# ------------------------------

# Extrapolation

def _partition(xsamp, xmin, xmax):
    """Return partitions of sample x values, given regression x values."""

    return (
        xsamp[xsamp < xmin],
        xsamp[(xsamp >= xmin) & (xsamp <= xmax)],
        xsamp[xsamp > xmax]
    )


def _slope_mean(point, data):
    """Return the mean slope of the given data."""

    # weights = np.exp(-(data[:,0] - point[0])**2 / 10.**2)

    der = misc.derivative(data[:, 0], data[:, 1])
    return np.average(der[~np.isnan(der)])

    return np.average(misc.derivative(data[:, 0], data[:, 1]))


def _line(point, slope):
    """Return a poly1d line object given a point and a slope."""

    return np.poly1d((slope, point[1] - slope * point[0]))


def _line_low(regression, nextrap=NEXTRAP):
    """Return a poly1d line object to extrapolate below the min x value."""

    return _line(
        regression[0],
        _slope_mean(regression[0], regression[:nextrap])
    )


def _line_high(regression, nextrap=NEXTRAP):
    """Return a poly1d line object to extrapolate above the max x value."""

    return _line(
        regression[-1],
        _slope_mean(regression[-1], regression[-nextrap:])
    )


def _interp_middle(regression):
    """Return an interp1d object to interpolate within data x bounds."""

    return interpolate.interp1d(
        regression[:, 0], regression[:, 1], copy=False
    )


# ------------------------------

# Exceptions

class LowessNotEnoughFitData(ValueError):
    pass


class LowessMismatchedInputSizes(ValueError):
    pass

# ------------------------------


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

    def _check_fitted(self):
        """Raise an exception if regression data is not available."""

        if not hasattr(self, "regression"):
            raise AttributeError("Model has not been trained yet.")

    def _check_extrap(self, xsmin, xsmax, max_extrap):
        """Raise an exception is extropolation is out of bounds."""

        if (
            ((xsmin - self.xmin) < -max_extrap) or
            ((xsmax - self.xmax) > max_extrap)
        ):
            raise ValueError(
                "Sample limits ({},{}) exceed the allowed extropolation ({}) "
                "from the data limits ({},{}).".format(
                    xsmin, xsmax, max_extrap,
                    round(self.xmin, 2), round(self.xmax, 2)
                )
            )

    def predict(self, x, max_extrap=None):
        """Obtain interpolated/extrapolated predictions from fitted regression.

        Parameters
        ----------
        x: array-like
            New sample of predictor values
        max_extrap [None]: float
            Maximum extrapolation range from the min and max x values of the
            fitted regression

        Returns
        -------
        numpy.ndarray of prediction values
        """

        self._check_fitted()

        x = _flat1d(x)

        if _any_nan(x):
            raise ValueError("the x array contains one or more NaNs")

        xargsort = x.argsort()

        if max_extrap is not None:
            self._check_extrap(x.min(), x.max(), max_extrap)

        # Sort regression by x values for easy head/tail extraction
        regr_xsorted = _sortby(self.regression, 0)

        # Partition data and interp/extrap
        (xlow, xmiddle, xhigh) = _partition(x[xargsort], self.xmin, self.xmax)
        (ylow, ymiddle, yhigh) = (np.array([]), np.array([]), np.array([]))
        if xlow.size > 0:
            ylow = _line_low(regr_xsorted)(xlow)
        if xmiddle.size > 0:
            ymiddle = _interp_middle(regr_xsorted)(xmiddle)
        if xhigh.size > 0:
            yhigh = _line_high(regr_xsorted)(xhigh)

        # Sorting undone to preserve input order
        return np.concatenate((ylow, ymiddle, yhigh))[xargsort.argsort()]

    def residual_percentile(self, per):
        """Return the given percentile of the residuals."""

        return np.percentile(self.residual, per)

    def predict_percentile(self, x, per, max_extrap=None):
        """Return the predictions corresponding to the to given residual
        percentile.

        See the documentation for LOWESS.predict for more details.
        """

        x = _flat1d(x)
        per = _flat1d(per)

        pred = (
            self.predict(x, max_extrap=max_extrap) +
            self.residual_percentile(per).reshape(per.size, -1)
        )

        if per.size == 1:
            if x.size == 1:
                return pred[0][0]
            else:
                return pred[0]
        else:
            return pred


# ------------------------------

def model(xtrn, ytrn, scale):
    """Return the LOWESS model object for the given training data.
    """
    return LOWESS(scale=scale).fit(xtrn, ytrn)


def predict(
    xtst, xtrn, ytrn,
    scale=None, max_extrap=None, positive_only=False,
    it=2
):
    """Directly return predictions from a LOWESS fit.
    """

    xtrn = _flat1d(xtrn)
    ytrn = _flat1d(ytrn)
    if positive_only:
        pos = ytrn > 0.
        xtrn = xtrn[pos]
        ytrn = ytrn[pos]

    return LOWESS(scale=scale, it=it).fit(xtrn, ytrn).predict(
        xtst, max_extrap=max_extrap
    )
