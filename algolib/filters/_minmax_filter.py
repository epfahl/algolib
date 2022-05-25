"""
Implementation of the van Herk-Gil-Werman minimum filter algorithm.
"""

import numpy as np

is_numba = True
try:
    import numba
except ImportError:
    print("WARNING: Module 'numba' could not be imported.")
    is_numba = False


def _border_values(data, mode='nearest', cval=0.0):
    """
    Given the border mode, return the border values.
    """

    if mode == 'nearest':
        left_value = data[0]
        right_value = data[-1]
    elif mode == 'constant':
        left_value = cval
        right_value = cval
    else:
        raise ValueError("Border mode '{}' not supported.".format(mode))

    return left_value, right_value


def _pad_data(data, size, left_value, right_value):
    """
    Pad the data on the left and right with the given border values.
    """

    ndata = data.size
    sm1 = size - 1
    data_padded = np.empty(ndata + 2 * sm1)
    data_padded[sm1:ndata + sm1] = data
    data_padded[:sm1] = left_value
    data_padded[ndata + sm1:] = right_value

    return data_padded


def _min_core(data, size):
    """
    Implementation of the van Herk-Gil-Werman minimum/maximum filter algorithm.

    ToDo
    -----
    In this version, there is an extra operation in the loop that fills the
    left and right arrays.  This operation handles the partial segment at the
    end of the array.  One (potentially) clean approach to handle this problem
    is to pad the input array to ensure an integral number of segments.
    """

    ndata = data.size
    result = data.copy()
    left = np.empty(size)
    right = np.empty(size)
    sm1 = size - 1

    for j in xrange(sm1, size * (ndata // size), size):

        left[0] = data[j]
        right[0] = data[j]
        for i in xrange(1, size):
            left[i] = min(left[i - 1], data[j - i])
            right[i] = min(right[i - 1], data[min(j + i, ndata - 1)])

        result[j - sm1] = left[sm1]
        result[j] = right[sm1]
        for i in xrange(1, sm1):
            result[j + i - sm1] = min(left[sm1 - i], right[i])

    return result


def _max_core(data, size):
    """
    Implementation of the van Herk-Gil-Werman minimum/maximum filter algorithm.

    ToDo
    -----
    In this version, there is an extra operation in the loop that fills the
    left and right arrays.  This operation handles the partial segment at the
    end of the array.  One (potentially) clean approach to handle this problem
    is to pad the input array to ensure an integral number of segments.
    """

    ndata = data.size
    result = data.copy()
    left = np.empty(size)
    right = np.empty(size)
    sm1 = size - 1

    for j in xrange(sm1, size * (ndata // size), size):

        left[0] = data[j]
        right[0] = data[j]
        for i in xrange(1, size):
            left[i] = max(left[i - 1], data[j - i])
            right[i] = max(right[i - 1], data[min(j + i, ndata - 1)])

        result[j - sm1] = left[sm1]
        result[j] = right[sm1]
        for i in xrange(1, sm1):
            result[j + i - sm1] = max(left[sm1 - i], right[i])

    return result


if is_numba:
    _min_core_jit = numba.jit(_min_core)
    _max_core_jit = numba.jit(_max_core)


def _min_max(data, size, left_value, right_value, comp='min', jit=False):
    """
    Interface to the van Herk-Gil-Werman minimum filter algorithm.
    """

    data_padded = _pad_data(data, size, left_value, right_value)

    if jit is True and is_numba is True:
        if comp == 'min':
            return _min_core_jit(
                data_padded, size)[:data_padded.size - size + 1]
        if comp == 'max':
            return _max_core_jit(
                data_padded, size)[:data_padded.size - size + 1]
    else:
        if comp == 'min':
            return _min_core(data_padded, size)[:data_padded.size - size + 1]
        if comp == 'max':
            return _max_core(data_padded, size)[:data_padded.size - size + 1]


def _minmax_filter(
    data, size, origin=0, mode='nearest', cval=0.0, comp='min', jit=False
):
    """
    Adapter function for the one-dimensional minimum/maximum filter using the
    van Herk-Gil-Werman algorithm.

    Parameters
    ----------
    data: array-like
        Input data
    size: int
        Length of the sliding filter window
    origin: int
        Placement of the filter relative to the filter center
    mode: ('constant', 'nearest')
        Determines how borders are handled
    cval: float
        Value assumed outside the data borders when mode='constant'
    comp: {'min', 'max'}
        Type of filter/comparison
    jit: bool
        If True, compile core filter function using numba.jit
    """

    if mode not in ('nearest', 'constant'):
        raise ValueError(
            "Boundary mode '{}' not supported; "
            "choose from ('nearest', 'constant')."
            .format(mode)
        )

    if size < 1:
        raise ValueError("Filter size must be >= 1.")

    if jit is True and is_numba is False:
        print("WARNING: numba.jit is not available; setting jit = False.")
        jit = False

    data = np.asarray(data)

    left_value, right_value = _border_values(data, mode=mode, cval=cval)
    min_max_output = _min_max(
        data, size, left_value, right_value, comp=comp, jit=jit)

    result = data.copy()
    first = size - 1 - (size // 2) - origin
    ndata = data.size
    if 0 <= first <= size - 1:
        result = min_max_output[first:ndata + first]
    else:
        if first < 0:
            result[:-first] = left_value
            result[-first:] = min_max_output[:ndata + first]
        else:
            delta = first - size + 1
            result[-delta:] = right_value
            result[:-delta] = min_max_output[first:]

    return result


docstr = """
    Calculate the one-dimensional {minmax} filter using the van Herk-Gil-Werman
    algorithm.

    Parameters
    ----------
    data: array-like
        Input data
    size: int
        Length of the sliding filter window
    origin: int
        Placement of the filter relative to the filter center
    mode: ('constant', 'nearest')
        Determines how borders are handled
    cval: float
        Value assumed outside the data borders when mode='constant'
    jit: bool
        If True, compile core filter function using numba.jit
"""


def minimum_filter(
    data, size, origin=0, mode='nearest', cval=0.0, jit=False
):

    return _minmax_filter(
        data, size, origin=origin, mode=mode, cval=cval, comp='min', jit=jit)


def maximum_filter(
    data, size, origin=0, mode='nearest', cval=0.0, jit=False
):

    return _minmax_filter(
        data, size, origin=origin, mode=mode, cval=cval, comp='max', jit=jit)


minimum_filter.__doc__ = docstr.format(minmax='minimum')
maximum_filter.__doc__ = docstr.format(minmax='maximum')
