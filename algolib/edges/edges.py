"""Find edges in 1D data.
"""

import numpy as np
import pandas as pd

from ..segments import segments


def _changes(trace, segs):
    return np.diff(trace[segs], axis=1).ravel()


def _segments(trace, diff_min):
    diff = np.ediff1d(trace)
    return np.concatenate([
        segments.segment_indices(diff >= diff_min),
        segments.segment_indices(diff <= -diff_min)])


def detect(trace, diff_min):
    """Return a DataFrame of edge index segments and changes, given a 1D
    trace.

    Parameters
    ----------
    trace: 1D array-like
        Input sequence
    diff_min: float
        Minimum adjacent difference magnitude to be included in an edge

    Returns
    -------
    pandas.DataFrame with columns
        'start': first index of edge
        'end': last index of edge
        'change': change in data value over the edge
    """
    trace = np.ravel(np.asarray(trace, dtype=float))
    segs = _segments(trace, diff_min)
    chngs = _changes(trace, segments)
    if len(segments) == 0:
        return pd.DataFrame(columns=['start', 'end', 'change'])
    return pd.DataFrame({
        'start': segs[:, 0],
        'end': segs[:, 1],
        'change': chngs})[
            ['start', 'end', 'change']].sort('start').reset_index(drop=True)
