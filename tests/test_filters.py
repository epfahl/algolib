"""Unit test for filter functions.
"""

import pytest
import numpy as np

from algolib import filters
from . import utils


@pytest.fixture
def arr1():
    a = np.ones(20)
    a[0] = -1
    a[5:10] = -1
    a[13] = -1
    a[15] = -1
    a[-2:] = -1
    return a


def check_expansion_start(ea):
    assert ea[0] == -1


def check_expansion_mid(ea, nmax):
    if nmax > 5:
        assert np.all(ea[1:-2] == 1)
    else:
        assert np.all(ea[5:10] == -1)
        assert np.all(ea[10:-2] == 1)


def check_expansion_end(ea):
    assert np.all(ea[-2:] == 1)


def check_arr1(fn, nmax, arr1):
    ea = fn(arr1, nmax)
    check_expansion_start(ea)
    check_expansion_mid(ea, nmax)
    check_expansion_end(ea)


@pytest.fixture
def arr2():
    npts = 3600
    r = np.random.random(npts)

    # insert "missing" segments
    r[100:230] = -1.0
    r[300:400] = -1.0
    r[1000:2000] = -1.0

    return r


def check_arr2(fn, arr2):
    raw = arr2
    expanded = fn(raw.copy(), nmax=120)

    # gap larger than 120
    assert raw[100] == expanded[100]

    # gap less than 120
    assert raw[300] != expanded[300]
    assert raw[299] == expanded[300]


def test_envelope_smooth():
    assert utils.array_equal(
        filters.envelope_smooth([1, 1, 2, 1, 0, 0, -1, 0], 2),
        np.array([1., 1., 1., 1., 0., 0., 0., 0.]))
    assert utils.array_equal(
        filters.envelope_smooth([1, 1, 1, 1, 0, 0, 0, 0], 2),
        np.array([1., 1., 1., 1., 0., 0., 0., 0.]))


def test_flatten_tail():
    assert utils.array_equal(
        filters.flatten_tail([1, 1, 0, 1, 1]),
        np.array([1, 1, 0, 0, 0]))


def test_flatten_head():
    assert utils.array_equal(
        filters.flatten_head([1, 1, 0, 1, 1]),
        np.array([0, 0, 0, 1, 1]))


def test_lowpass_ends():
    """Test begninning, end of signal
    """
    trace = np.zeros(20)
    trace[:5] = 1
    trace[-5:] = -0.5

    btrace3 = filters.lowpass(trace, tmin=3)
    btrace10 = filters.lowpass(trace, tmin=10)

    assert all((
        (btrace3[0] == trace[0]),
        (btrace3[-1] == trace[-1]),
        (btrace10[0] == trace[0]),
        (btrace10[-1] == trace[-1])
    ))


def test_highpass():
    """Test highpass filter
    """
    trace = np.zeros(70)
    ndx0 = np.arange(5, 10)
    ndx1 = np.arange(20, 30)
    ndx2 = np.arange(40, 60)

    trace[ndx0] = 1
    trace[ndx1] = 1
    trace[ndx2] = 1

    btrace10 = filters.bandpass_preserve(trace, tmax=11)
    htrace10 = filters.highpass(trace, tmax=11)

    assert all((
        np.all(btrace10 == htrace10),
        np.all(btrace10[ndx0] == 1),
        np.all(btrace10[ndx1] == 1),
        np.all(btrace10[ndx2] == 0)))


def test_lowpass():
    """Test lowpass filter
    """
    trace = np.zeros(100)
    ndx0 = np.arange(5, 10)
    ndx1 = np.arange(20, 30)
    ndx2 = np.arange(40, 60)

    trace[ndx0] = 1
    trace[ndx1] = 1
    trace[ndx2] = 1

    btrace10 = filters.bandpass_preserve(trace, tmin=10)
    ltrace10 = filters.lowpass(trace, tmin=10)

    assert all((
        np.all(btrace10 == ltrace10),
        np.all(btrace10[ndx0] == 0),
        np.all(btrace10[ndx1] == 1),
        np.all(btrace10[ndx2] == 1)))


def test_bandpass_preserve_missing():
    """Test filter with missing data
    """
    trace = np.zeros(30)
    trace[5:10] = 1
    trace[10:15] = -1
    trace[15:20] = 1

    btrace3 = filters.bandpass_preserve(trace, tmin=3)
    btrace10 = filters.bandpass_preserve(trace, tmin=10)
    btrace20 = filters.bandpass_preserve(trace, tmin=20)

    assert all((
        np.all(btrace3[5:20] == trace[5:20]),
        np.all(btrace10[5:20] == trace[5:20]),
        np.all(btrace20[10:15] == -1),
        np.all(btrace20[:10] == 0),
        np.all(btrace20[15:] == 0)))


def test_bandpass_negative():
    """Test filter where some data is negative
    """
    trace = np.zeros(30)

    trace[5:25] = -0.5

    ndx0 = np.arange(10, 12)
    trace[ndx0] = 1.0
    ndx1 = np.arange(15, 20)
    trace[ndx1] = 1.0

    ltrace3 = filters.lowpass(trace, tmin=3)
    htrace3 = filters.highpass(trace, tmax=3)

    assert all((
        np.all(ltrace3[ndx0] == -0.5),
        np.all(htrace3[ndx0] == 1.0),
        np.all(ltrace3[ndx1] == 1.0),
        # positive segment is longer than tmax for highpass
        np.all(htrace3[ndx1] == 0),
    ))


def test_fill_mask_modes():
    """Exercise the different modes of the fill_mask function.
    """

    trace = np.array([-1, -1, 1, 2, -1, -1, 5, 6, -1, -1])
    mask = (trace == -1)

    ftrace_left = filters.fill_mask(trace, mask, mode='holdleft')
    ftrace_right = filters.fill_mask(trace, mask, mode='holdright')
    ftrace_interp = filters.fill_mask(trace, mask, mode='interp')

    assert all([
        np.all(ftrace_left == np.array([1, 1, 1, 2, 2, 2, 5, 6, 6, 6])),
        np.all(ftrace_right == np.array([1, 1, 1, 2, 5, 5, 5, 6, 6, 6])),
        np.all(ftrace_interp == np.array([1, 1, 1, 2, 3, 4, 5, 6, 6, 6]))])


def test_fill_mask_all():
    """Test the output of fill_mask when all or none of data is missing.
    """

    trace_all = -np.ones(10)
    trace_none = np.ones(10)

    ftrace_all = filters.fill_mask(trace_all, trace_all == -1)
    ftrace_none = filters.fill_mask(trace_none, trace_none == -1)

    assert all([
        np.all(ftrace_all == trace_all),
        np.all(ftrace_none == trace_none)])


def test_envelope_filter():

    trace = np.zeros(200)
    trace[50:150] = 1.
    trace_frac = (trace > 0.).mean()

    ftrace_low = filters.envelope(trace, bw=100, mode='min')
    ftrace_frac_low = (ftrace_low > 0.).mean()

    ftrace_high = filters.envelope(trace, bw=101, mode='min')
    ftrace_frac_high = (ftrace_high > 0.).mean()

    assert (
        (trace_frac == 0.5) &
        (ftrace_frac_low == 0.5) &
        (ftrace_frac_high == 0.))


def test_fill_missing():

    trace = np.ones(15)
    trace[10:] = 2
    trace[:3] = -1
    trace[-3:] = -1
    trace[5:10] = -1
    trace[10:] = 2

    trace_hold_3 = filters.fill_missing(trace, nmax=3, mode='hold')
    trace_hold_5 = filters.fill_missing(trace, nmax=5, mode='hold')
    trace_interp = filters.fill_missing(trace, mode='interp')

    nmissing_3 = np.count_nonzero(trace_hold_3 == -1)
    nmissing_5 = np.count_nonzero(trace_hold_5 == -1)
    spot_value = trace_interp[7]

    assert all([nmissing_3 == 5, nmissing_5 == 0, spot_value == 1.5])
