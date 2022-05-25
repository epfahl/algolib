"""
Module to support a piece-wise linear function (with power-law extrapolation)
that maps (-inf, inf) or [0, inf) to [0, 1].
"""

import numpy as np

from . import pcinterp

ANCHORS_REAL_INCR = ((-100., 0.2), (100., 0.8))
ANCHORS_POS_DECR = ((100., 0.2),)
PEXTRAP = 1.


# ------------------------------

# Functions for computing model parameter values and extrapolation

def _a_hi(anchor, slope, p):
    return (p / slope)**p * (1. - anchor[1])**(p + 1)


def _b_hi(anchor, slope, p):
    return anchor[0] - p * (1. - anchor[1]) / slope


def _a_lo(anchor, slope, p):
    return (p / slope)**p * anchor[1]**(p + 1)


def _b_lo(anchor, slope, p):
    return anchor[0] + p * anchor[1] / slope


def _slope(alo, ahi):
    return (ahi[1] - alo[1]) / (ahi[0] - alo[0])


def _hi_extrap(x, a, b, p):
    return 1. - a * (x - b)**(-p)


def _lo_extrap(x, a, b, p):
    return a * (b - x)**(-p)


# ------------------------------

# Public functions for score mapping

def score_real_incr(x, anchors=ANCHORS_REAL_INCR, p_extrap=PEXTRAP):
    """Monotonically increasing function that maps the vector of input values in
    (-inf, inf) to a vector of values in (0, 1) using piece-wise linear
    interpolation between anchor points and power-law extrapolation outside of
    the anchors.

    Parameters
    ----------
    x: array-like of floats
        Input scalars in (-inf, inf)
    anchors: Nx2 array-like
        Discrete anchor points for mapping; monotonically increasing sequence
    p_extrap:
        Power used for hyperbolic extrapolation outside of the anchors

    Returns
    -------
    score: numpy.ndarray of floats
    """

    anch = pcinterp.validate_anchors(anchors, test_mono_incr=True)

    shi = _slope(anch[-2], anch[-1])
    if p_extrap is None:
        hiextrap = None
    else:
        hiextrap = lambda x: _hi_extrap(
            x,
            _a_hi(anch[-1], shi, p_extrap),
            _b_hi(anch[-1], shi, p_extrap),
            p_extrap
        )

    slo = _slope(anch[0], anch[1])
    if p_extrap is None:
        loextrap = None
    else:
        loextrap = lambda x: _lo_extrap(
            x,
            _a_lo(anch[0], slo, p_extrap),
            _b_lo(anch[0], slo, p_extrap),
            p_extrap
        )

    return pcinterp.pcinterp1d(
        x,
        anchors=anchors,
        hiextrap=hiextrap,
        loextrap=loextrap,
    )


def score_pos_decr(x, anchors=ANCHORS_POS_DECR, p_extrap=PEXTRAP):
    """Monotonically decreasing function that maps a vector of input values in
    (0, inf) to a vector of values in (0, 1) using piece-wise linear
    interpolation between anchor points and power-law extrapolation outside of
    the anchors.

    Parameters
    ----------
    x: array-like of floats
        Input scalars in (0, inf)
    anchors: Nx2 array-like
        Discrete anchor points for mapping; monotonically decreasing sequence
    p_extrap:
        Power used for hyperbolic extrapolation beyond the anchor with the
        largest x value

    Returns
    -------
    score: numpy.ndarray of floats
    """

    anch = pcinterp.validate_anchors(anchors, test_mono_decr=True)

    # Transform the anchors to be compatible with score_real_incr
    anchtr = anch.copy()
    anchtr[:, 1] *= 0.5
    anchtr[:, 0] *= -1.
    root_anch = (0., .5)
    anchtr = np.concatenate((anchtr, [root_anch]))

    return 2. * score_real_incr(-x, anchors=anchtr, p_extrap=PEXTRAP)
