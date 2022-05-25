"""Uncategorized utilities.
"""

import numpy as np
import functools
import itertools


def sloppy_intersect(v1, v2, slop=1):
    """Given two vectors of unique integers, generate masks that indicate when
    a value is present in both vectors, within some slop.

    Parameters
    ----------
    v1, v2: array-like (1D)
        Input vectors of unique integers
    slop: int
        Integer slop

    Returns
    -------
    mask1, mask2: numpy.ndarray
        Boolean vectors that indicate matches

    Notes
    -----
    This function effectively generalizes numpy.in1d.

    Examples
    --------
    >>> v1 = np.array([1, 10, 20])
    >>> v2 = np.array([3, 10, 21, 17])
    >>> mask1, mask2 = sloppy_intersect(v1, v2)
    >>> v1[mask1]
    array([10, 20])
    >>> v2[mask2]
    array([10, 21])
    """

    in1d = functools.partial(np.in1d, assume_unique=True)

    njitters = 2 * slop + 1
    jitters = np.arange(-slop, slop + 1).reshape(njitters, 1)
    v1jit = jitters + np.tile(v1, (njitters, 1))

    mask1 = np.vstack([in1d(v1j, v2) for v1j in v1jit]).any(axis=0)
    mask2 = np.vstack([in1d(v2, v1j) for v1j in v1jit]).any(axis=0)

    return mask1, mask2


def jitter_triangular(
    array,
    half_width=None,
    mask=None,
    random_state=None
):
    """To each valid element of the given array, add a random perturbation
    from a triangular distribution.

    Parameters
    ----------
    array: numpy.ndarray
        Input data
    half_width: float/int
        Half of the full width of the symmetric triangular distribution
    mask: numpy.ndarray, bool
        Boolean mask that indicates which values should be jittered
    random_state: instance of numpy.random.RandomState
        If None, the global random state is called

    Returns
    -------
    array: numpy.ndarray
        Jittered array

    Examples
    --------
    >>> array = np.array([1., 10.])
    >>> jitter_triangular(array, half_width=1)
    array([  0.60972906,  10.28306443])
    """

    if half_width is None or half_width == 0:
        return array

    if random_state is None:
        tri = np.random.triangular
    else:
        tri = random_state.triangular

    if mask is None:
        mask = np.ones(array.shape, dtype=bool)

    array_jittered = array.copy()
    array_jittered += tri(
        -half_width,
        0.,
        half_width,
        size=array.shape
    )
    array_jittered[~mask] = array[~mask]

    return array_jittered


def bimodal_partition(data, sorted=False):
    """
    Find the partition value that maximizes the inter-class variance of a
    bimodal distribution.

    Parameters
    ----------
    data: numpy.ndarray
        1D input array
    sorted: bool
        If True, assume the input data is sorted; sort otherwise

    Returns
    -------
    part: float
        Partition value

    Notes
    -----
    This is an adaptation of Otsu's method for finding an optimal partition of
    quantized intensity data.  This is a popular technique for adaptive
    foreground/background segmentation in image processing.

    Examples
    --------
    >>> data = np.array([-10, 10])
    >>> bimodal_partition(data)
    0.0
    >>> np.random.seed(1234)
    >>> g1 = np.random.randn(100) - 10.
    >>> g2 = np.random.randn(100) + 10.
    >>> data = np.concatenate((g1, g2))
    >>> bimodal_partition(data)
    0.12155273890971152  # expect (-0.5, 0.5) for different seeds
    """

    # Sort data if not already sorted
    if sorted is False:
        data = np.sort(data)

    # Precompute the cumulants and population counts
    cum = data.cumsum()
    num = np.arange(1, data.size)

    # Compute the objective function for all possible partitions
    obj = cum[:-1]**2 / num + (cum[-1] - cum[:-1])**2 / num[::-1]

    # Find the index and the data value at which the vector objective function
    # is a maximum
    ipart = obj.argmax()
    part = 0.5 * (data[ipart] + data[ipart + 1])

    return part


def zerocross_index_pairs(rel):
    """Return the pairs of indices that straddle each zero crossing.
    """
    idx = np.diff((rel > 0).astype(int)).nonzero()[0]
    return np.array([idx, idx + 1]).T


def zerocross_index_nearest(rel):
    """Return the indices nearest to each zero crossing.
    """
    pairs = zerocross_index_pairs(rel)
    return np.choose(np.abs(rel[pairs]).argmin(axis=1), pairs.T)


def levelcross_index(trace, level):
    """Return incides and diffs each place a sequence crosses a given level.

    Parameters
    ----------
    trace: numpy.ndarray
        1D input array
    level: float
        Value checked for crossing

    Returns
    -------
    List of crossing indices (before crossing) and gradients
    """
    rel = trace - level
    idx_cross = np.diff((rel > 0).astype(int)).nonzero()[0]
    return idx_cross, np.diff(trace)[idx_cross]


def levelcross_interp(x, y, level):
    """Return the x values and slopes for each level crossing.  Each x value
    is determined by interpolating between nearest data points.

    Parameters
    ----------
    x, y: array-like
        1D x and y input data
    level: float
        y level

    Returns
    -------
    List of interpolated x crossing values and derivatives.
    """

    x, y = map(np.atleast_1d, (x, y))
    if x.size != y.size:
        return ValueError("Input array must have the same length.")
    ipairs = zerocross_index_pairs(y - level)
    xpairs = x[ipairs]
    ypairs = y[ipairs]
    fcross = (level - ypairs[:, 0]) / np.diff(ypairs).ravel()
    dx = np.diff(xpairs).ravel()
    return (
        x[ipairs][:, 0] + (x[1] - x[0]) * fcross,
        np.diff(y)[ipairs[:, 0]] / dx)


def lazy_join(sets, min_match=None):
    """Recursively merge sets with finite intersection.

    Parameters
    ----------
    sets: list of sets
        Input sets
    min_match: int
        Minimum number of matches required to join two sets

    Returns
    -------
    sets: list of sets
        Final list of disjoint sets

    Notes
    -----
    The premise here is that each pair of objects in the same set are similar
    according to some metric, and that this similarity is transitive.  It is
    the assumption of transitivity that gives this operation its 'lazy'
    character.

    Examples
    --------
    >>> sets = [{1, 2}, {1, 3}, {5, 6}]
    >>> lazy_join(sets)
    [{1, 2, 3}, {5, 6}]
    """

    if min_match is None:
        min_match = 1

    nsets = len(sets)
    is_merger = False
    for i, j in itertools.combinations(range(nsets), 2):
        si = sets[i]
        sj = sets[j]
        if len(si & sj) >= min_match:
            is_merger = True
            sets[i] = si.union(sj)
            del sets[j]
            break

    if is_merger:
        sets = lazy_join(sets, min_match=min_match)

    return sets


def window_correlate(a, b, window_max=1800):
    """
    Cross-correlate two equal-length sequences within a finite range of shifts.

    Parameters
    ----------
    a, b: 1D ndarrays
        Inupt vectors; equal length
    window_max: int
        Max absolute data shift in sample units

    Returns
    -------
    corr: ndarray
        Correlation result

    ToDo
    ----
    * Is there an efficient way to do the loop over shifts via broadcasting?
      It bothers me that this isn't easily vectorizable.
      (Yes, do it with multiplication by a circulant matrix.)
    * Use truncated norm for each shift, instead of a single global
      standardization.
    """

    def _standardize(v):
        """Standardize the given 1D ndarray.
        """
        std = v.std()
        if std == 0.:
            std = 0.01
        return (v - v.mean()) / (std * np.sqrt(float(v.size)))

    if len(a) != len(b):
        raise ValueError(
            "Sequences being correlated must have the same length.")

    # Transform data vectors to zero mean and unit variance
    at = _standardize(a)
    bt = _standardize(b)

    # Pearson correlation for each shift
    window_vec = range(1, window_max + 1)
    c0 = np.dot(at, bt)
    cm = [np.dot(bt[s:], at[:-s]) for s in window_vec]
    cp = [np.dot(at[s:], bt[:-s]) for s in window_vec]

    corr = cm[::-1] + [c0] + cp

    return np.array(corr)


def mlshift(
        a, b,
        window_max=900,
        trend_window=300,
        trend_percentile=50
):
    """Compute and measure the significance of the most likely shift between
    two equal-length sequences.

    Parameters
    ----------
    a, b: 1D ndarrays
        Input vectors; equal length
    window_max: int
        Max absolute data shift in sample units (see window_correlate.__doc__)
    trend_window, trend_percentile: int, float
        See trend.__doc__

    The sign of the shift is interpreted as follows.

    < 0: B LAGS A
    --------
    A:    ---++---+++--++++--
    B:    -----++---+++--++++

    > 0: B LEADS A
    ---------
    A:    ---++---+++--++++--
    B:    -++---+++--++++----
    """

    # Pearson cross correlation, index of peak, and shift value
    corr = window_correlate(a, b, window_max)
    idxmax = np.argmax(corr)
    shift = idxmax - window_max
    ccf = corr[idxmax]

    # Detrend the correlation and compute the median-based Z score at the peak
    detrend = corr - trend(
        corr,
        window=trend_window, percentile=trend_percentile, sigma_ratio=0.25
    )
    zmax = median_zscore(detrend)[idxmax]

    return shift, ccf, zmax


def pad_ends(a, mode='copy', constant=0):
    """Pad the input array at each end with a single values.  The length of
    the return array is 2 more than the input array.

    Parameters
    ----------
    a: array-like (1D)
        Input array
    mode: str
        'copy' -- Copy the end values to the pad entries
        'constant' -- Pad the ends with a constant values
    constant: float/int
        Value appended to the ends when mode='constant'

    Returns
    -------
    Padded array

    Examples
    --------
    >>> a = np.arange(5)
    >>> pad_ends(a, mode='copy')
    array([0, 0, 1, 2, 3, 4, 4])
    >>> pad_ends(a, mode='constant', constant=-1)
    array([-1,  0,  1,  2,  3,  4, -1])
    """

    if mode == 'copy':
        return np.lib.pad(a, (1, 1), mode='symmetric')
    elif mode == 'constant':
        return np.lib.pad(
            a, (1, 1), mode='constant', constant_values=(constant, constant))
    else:
        raise ValueError(
            "mode '{}' not recognized; must be 'copy' or 'constant'")


def derivative(x, y, is_sorted=False):
    """
    Return a vector of numerical approximations to dy/dx, given 1D sampled
    data, where the x values are not necessarily uniformly sampled.
    """

    x = np.ravel(np.asarray(x, dtype=float))
    y = np.ravel(np.asarray(y, dtype=float))

    if not is_sorted:
        argsort = x.argsort()
        x = x[argsort]
        y = y[argsort]

    return np.gradient(y) / np.gradient(x)
