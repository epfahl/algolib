"""One-dimensional filters.
"""

import numpy as np
import functools
import scipy.ndimage as ndi
from scipy import interpolate

from ..segments import segments
from . import _minmax_filter


def minimum_filter(data, size, origin=0, mode='nearest', cval=0., jit=False):
    """
    Compute the sliding minimum filter.

    Parameters
    ----------
    trace: 1D numerical sequence
        Input data
    size: int
        Window size of the sliding min/max filters
    mode: str
        'min' => minimum envelope; 'max' => maximum envelope
    jit [False]: bool
        If True, use the fast O(N) versions of the min/max filters (runs
        numba.jit compiled code if numba is available).  If False, use the
        implementation in scipy.ndimage.

    Result
    ------
    result: 1D NumPy vector
        Filter result
    """
    if jit:
        return _minmax_filter.minimum_filter(
            data, size, origin=origin, mode=mode, cval=cval, jit=True)
    else:
        return ndi.minimum_filter1d(
            data, size, origin=origin, mode=mode, cval=cval)


def maximum_filter(data, size, origin=0, mode='nearest', cval=0., jit=False):
    """
    Compute the sliding maximum filter.

    Parameters
    ----------
    trace: 1D numerical sequence
        Input data
    size: int
        Window size of the sliding min/max filters
    mode: str
        'min' => minimum envelope; 'max' => maximum envelope
    jit [False]: bool
        If True, use the fast O(N) versions of the min/max filters (runs
        numba.jit compiled code if numba is available).  If False, use the
        implementation in scipy.ndimage.

    Result
    ------
    result: 1D NumPy vector
        Filter result
    """
    if jit:
        return _minmax_filter.maximum_filter(
            data, size, origin=origin, mode=mode, cval=cval, jit=True)
    else:
        return ndi.maximum_filter1d(
            data, size, origin=origin, mode=mode, cval=cval)


def envelope(trace, bw=1, mode='min', jit=False):
    """
    Compute the min or max envelope of the given function.

    Parameters
    ----------
    trace: 1D numerical sequence
        Input data
    bw: int
        Window size of the sliding min/max filters
    mode: str
        'min' => minimum envelope; 'max' => maximum envelope
    jit [False]: bool
        If True, use the fast O(N) versions of the min/max filters (runs
        numba.jit compiled code if numba is available)

    Result
    ------
    result: 1D NumPy vector
        Filter result
    """

    if bw == 1:
        return trace

    minfilter1d = functools.partial(
        minimum_filter, size=bw, mode='nearest', jit=jit)
    maxfilter1d = functools.partial(
        maximum_filter, size=bw, mode='nearest', jit=jit)

    orgright = (bw - 1) // 2
    orgleft = -(bw // 2)

    if mode == 'min':

        first = minfilter1d
        second = maxfilter1d
        third = np.maximum

    elif mode == 'max':

        first = maxfilter1d
        second = minfilter1d
        third = np.minimum

    result = first(trace, origin=orgleft)
    result = second(result, origin=orgright)
    result_fix = first(trace[:bw], origin=orgright)
    result_fix = second(result_fix, origin=orgleft)
    result[:bw] = third(result_fix, result[:bw])

    return result


def envelope_smooth(trace, size):
    """Smooth the trace by composing min and max envelope filters with the
    given size.

    Examples
    --------
    >>> envelope_smooth([1, 1, 2, 1, 0, 0, -1, 0], 2)
    array([1., 1., 1., 1., 0., 0., 0., 0.])
    >>> envelope_smooth([1, 1, 1, 1, 0, 0, 0, 0])
    array([1., 1., 1., 1., 0., 0., 0., 0.])
    """

    def env(t, m):
        return envelope(t, size, m)

    return 0.5 * (
        env(env(trace, 'min'), 'max') +
        env(env(trace, 'max'), 'min'))


def fill_mask(trace, mask, nmax=np.inf, mode='hold'):
    """
    Given a trace and a mask, fill mask == True segments with a neighbor value
    or with linear interpolation.

    Parameters
    ----------
    trace: 1D array-like
        Raw input data
    nmax: int or np.inf
        Segments with more than nmax points are not filled.
    mode: str
        Method used to fill segemnts.
        'holdleft'/'holdright' -- hold value to left/right of missing segment
        'hold' -- alias of 'holdleft'
        'interp' -- linear interpolation across missing data

    Returns
    -------
    trace_filled: numpy.ndarray
        Trace with filled mask == True segments
    """

    # Do nothing if either no data is missing or all data is missing
    if np.all(mask) or np.all(~mask):
        return trace

    # Start/Stop indices of mask == True segments
    endpts = segments.segment_indices(mask)

    trace_filled = trace.copy()
    for lp, rp in endpts:

        # Don't fill gaps that are too large
        if (rp - lp) > nmax:
            continue

        if lp == 0:
            lval = trace[rp]
        else:
            lval = trace[lp - 1]

        if rp == trace.size:
            rval = trace[lp - 1]
        else:
            rval = trace[rp]

        if mode in ('hold', 'holdleft'):
            trace_filled[lp:rp] = lval

        elif mode in ('holdright'):
            trace_filled[lp:rp] = rval

        elif mode in ('interp'):
            trace_filled[lp:rp] = interpolate.interp1d(
                [lp - 1, rp], [lval, rval]
            )(np.arange(lp, rp))

    return trace_filled


def fill_missing(trace, nmax=np.inf, mode='hold'):
    """
    Fill segments of missing data with a neighbor value or with linear
    interpolation.

    See doc for plotwatt.util.scipyutils.fill_mask.
    """
    trace = np.array(trace)
    return fill_mask(trace, (trace == -1), nmax=nmax, mode=mode)


def fill_below(trace, val, nmax=np.inf, mode='hold'):
    """Fill segments of data with values less than or equal to the given value
    with a neighbor value or with linear interpolation.
    """
    trace = np.array(trace)
    return fill_mask(trace, (trace <= val), nmax=nmax, mode=mode)


def bandpass(
    trace,
    tmin=None,
    tmax=None,
    tbridge=None,
    sample_time=1,
    nonnegative=False,
):
    """
    Apply an envelope filter such that the resulting trace has up-down
    fluctuation times in the interval [tmin, tmax].

    Parameters
    ----------
    trace: array-like (Nx1)
        Input data
    tmin [None]: int
        Minimum up-down fluctuation time
    tmax [None]: int
        Maximum up-down fluctuation time
    tbridge [None]: int
        Short dips of duration <= tbridge are bridged over
    sample_time [1]: int
        Sample time of the data in seconds
    nonnegative [False]: bool
        Set to True if we want to prevent the filter from adding energy

    Returns
    -------
    btrace: numpy.ndarray
        Bandpass filtered data
    """

    def lowp(x, t):
        return envelope(x, bw=t // sample_time, mode='min', jit=True)

    if tmin < sample_time:
        tmin = None
    if tmax < sample_time or tmax > len(trace):
        tmax = None

    if (tmin is None) and (tmax is None):
        return trace

    btrace = trace.copy()
    if tmin is not None:
        btrace = lowp(btrace, int(tmin))
    if tmax is not None:
        lp = lowp(btrace, int(tmax))
        if nonnegative:
            lp[lp < 0] = 0
        btrace -= lp

    if tbridge:
        btrace = envelope(btrace, bw=int(tbridge) // sample_time, mode='max')

    return btrace


def bandpass_preserve(
    trace, tmin=None, tmax=None, tbridge=None, sample_time=1,
    missing_sentinel=-1
):
    """
    Bandpass filter that preserves missing and negative data. The latter is
    important if we are decomposing a signal by energy, as it will not result
    in /adding/ energy.

    See doc for plotwatt.util.scipyutils.bandpass.
    """

    missing_mask = (trace == missing_sentinel)
    trace_held = fill_mask(trace, missing_mask)
    btrace = bandpass(
        trace_held, tmin, tmax, tbridge, sample_time, nonnegative=True
    )
    btrace[missing_mask] = -1

    return btrace


def lowpass(trace, tmin, tbridge=None, sample_time=1):
    """
    Convenience function of lowpass filtering.

    See doc for bandpass.

    Notes
    -----
    * Passes parts of the signal with up-down fluctuations longer than tmin.
    * Skips over brief dips of duration tbridge.
    """

    return bandpass_preserve(
        trace, tmin=tmin, tmax=None, tbridge=tbridge, sample_time=sample_time)


def highpass(trace, tmax, tbridge=None, sample_time=1):
    """
    Convenience function for highpass filtering.

    See doc for bandpass.

    Notes
    -----
    * Passes parts of the signal with up-down fluctuations shorter than tmax.
    * Skips over brief dips of duration tbridge.
    """

    return bandpass_preserve(
        trace, tmin=None, tmax=tmax, tbridge=tbridge, sample_time=sample_time)


def uniform_filter(trace, size, origin='right'):
    """
    Convenience function to access ndimage.uniform_filter1d with allowed
    origins.

    Parameters
    ----------
    trace: 1D array-like
        Input data
    size: int
        Number of samples in window
    origin: str
        'left' -- origin at left edge of window
        'center' -- origin at center of window
        'right' -- origin at right edge of window

    Returns
    -------
    numpy.ndarray
        Filtered array
    """

    if origin == 'center':
        origin = 0
    elif origin == 'left':
        origin = -(size // 2)
    elif origin == 'right':
        origin = origin = (size - 1) // 2
    else:
        raise ValueError(
            "origin '{}' not recognized; must be 'center', 'right', or 'left'"
            .format(origin))

    return ndi.uniform_filter1d(trace, size, origin=origin, mode='nearest')


def dog_filter(trace, sigma):
    """Derivitive-of-Gaussian filter, normalized so that the result has unit
    amplitude for a unit step.
    """

    return np.sqrt(2 * np.pi) * sigma * ndi.gaussian_filter1d(
        trace, sigma, order=1, mode='nearest')


def flatten_tail(trace):
    """For points at indices after the minimum, set the values to the minimum
    value.

    Example
    -------
    >>> flatten_tail([1, 1, 0, 1, 1])
    array([1, 1, 0, 0, 0])
    """
    ctrace = np.array(trace).copy()
    amin = ctrace.argmin()
    ctrace[amin:] = ctrace[amin]
    return ctrace


def flatten_head(trace):
    """For points at indices before the minimum, set the values to the minimum
    value.

    Example
    -------
    >>> flatten_head([1, 1, 0, 1, 1])
    array([0, 0, 0, 1, 1])
    """
    return flatten_tail(trace[::-1])[::-1]
