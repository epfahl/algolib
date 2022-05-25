"""
Module supporing linear interpolation and functional extrapolation on a
piecewise-continuous function.
"""

import functools

import numpy as np
from scipy import interpolate


# ------------------------------

def validate_anchors(anchors, test_mono_incr=False, test_mono_decr=False):

    a = np.atleast_2d(anchors)
    a = a[a[:, 0].argsort()]

    if a.shape[0] < 2:
        raise ValueError(
            "There must be at least 2 anchor points: {}.".format(a)
        )

    if test_mono_incr:
        if np.any(np.diff(a[:, 1]) < 0):
            raise ValueError(
                "The anchors are not consistent with a monotonically "
                "increasing function: {}.".format(a)
            )

    if test_mono_decr:
        if np.any(np.diff(a[:, 1]) > 0):
            raise ValueError(
                "The anchors are not consistent with a monotonically "
                "decreasing function: {}.".format(a)
            )

    return a


# ------------------------------

# Serial mapping
def _pcinterp1d_serial(
    x, anchors,
    hiextrap, loextrap, extrap_error
):
    """
    Parameters
    ----------
    anchors: Nx2 numpy.ndarray
        x-ordered array of anchor points
    """

    # Interior interpolation
    if anchors[0][0] <= x <= anchors[-1][0]:
        return interpolate.interp1d(anchors[:, 0], anchors[:, 1])(x)

    # Low extrapolation
    if x < anchors[0][0]:
        if not extrap_error:
            if loextrap is not None:
                return loextrap(x)
            else:
                return anchors[0][1]
        else:
            raise ValueError(
                "Cannot extrapolate below minimum anchor x value: {} < {}."
                .format(x, anchors[0][0])
            )

    # High extrapolation
    if x > anchors[-1][0]:
        if not extrap_error:
            if hiextrap is not None:
                return hiextrap(x)
            else:
                return anchors[-1][1]
        else:
            raise ValueError(
                "Cannot extrapolate above maximum anchor x value: {} > {}."
                .format(x, anchors[-1][0])
            )


# Vectorized mapping
_pcinterp1d_vector = np.vectorize(
    functools.partial(
        _pcinterp1d_serial,
        anchors=None,
        hiextrap=None,
        loextrap=None,
        extrap_error=False,
    ),
    excluded=[
        'anchors',
        'hiextrap', 'loextrap', 'extrap_error'
    ]
)


# ------------------------------

def pcinterp1d(
    x, anchors=None,
    hiextrap=None, loextrap=None, extrap_error=False,
    test_mono_incr=False, test_mono_decr=False
):
    """
    Parameters
    ----------
    x: float
        Input scalar in (-inf, inf)
    anchors: Nx2 array-like
        anchor points for piecewise-continuous mapping

    Returns
    -------
    numpy.ndarray of floats
    """

    anch = validate_anchors(
        anchors,
        test_mono_incr=test_mono_incr, test_mono_decr=test_mono_decr
    )

    return _pcinterp1d_vector(
        x, anchors=anch,
        hiextrap=hiextrap, loextrap=loextrap
    )
