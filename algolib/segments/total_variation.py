"""Piece-wise constant modeling of timeseries.

Notes
-----
* This is here largely for historical record.  This will probably be
  implemented differently if it is used in the future.
"""

# ------------------------------

import numpy as np

try:
    from numba import jit
except ImportError:
    print("WARNING: Module 'numba' could not be imported.")


# ------------------------------

# @jit(nopython=True)
def update_mean(cnt, mean, val):
    """Update the running mean."""

    return (cnt * mean + val) / (cnt + 1)


# @jit(nopython=True)
def _init_seg(i, x):

    return i, x, x, x, 1


# @jit(nopython=False)
def _tvcore(trace, tv_max, segment_indices, segment_means):

    n = trace.size
    sstart, smin, smax, smean, scnt = _init_seg(0, trace[0])
    iseg = 0

    for i, x in enumerate(trace[1:], 1):
        newmin = min(smin, x)
        newmax = max(smax, x)

        if ((newmax - newmin) <= tv_max):
            smin = newmin
            smax = newmax
            smean = update_mean(scnt, smean, x)

            if i == (n - 1):
                segment_indices[iseg, 0] = sstart
                segment_indices[iseg, 1] = i
                segment_means[iseg] = update_mean(scnt, smean, x)
                break

            scnt += 1

        else:
            segment_indices[iseg, 0] = sstart
            segment_indices[iseg, 1] = i - 1
            segment_means[iseg] = smean
            iseg += 1

            if i == (n - 1):
                segment_indices[iseg, 0] = i
                segment_indices[iseg, 1] = i
                segment_means[iseg] = x
                break

            sstart, smin, smax, smean, scnt = _init_seg(i, x)

    return segment_indices, segment_means, iseg


def compress_segments(segment_indices, segment_means):
    utimes, uindices = np.unique(segment_indices.ravel(), return_index=True)
    return utimes, np.repeat(segment_means, 2)[uindices]


def tvsegments(trace, tv_max):
    segment_indices, segment_means, ilast = _tvcore(
        trace, tv_max,
        np.empty((trace.size, 2), dtype=np.int64),
        np.empty(trace.size)
    )
    return segment_indices[:ilast + 1], segment_means[:ilast + 1]


def tvseries(trace, tv_max):
    return compress_segments(*tvsegments(trace, tv_max))
