# This file contains the codes from SciPy project.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from collections.abc import Callable
import functools
import math
import sys

import numpy as np

from optuna.samplers._tpe._erf import erf


_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def _log_sum(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    return np.logaddexp(log_p, log_q)


def _log_diff(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    return log_p + np.log1p(-np.exp(log_q - log_p))


@functools.lru_cache(1000)
def _ndtr_single(a: float) -> float:
    x = a / 2**0.5

    if x < -1 / 2**0.5:
        y = 0.5 * math.erfc(-x)
    elif x < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 1.0 - 0.5 * math.erfc(x)

    return y


def _ndtr(a: np.ndarray) -> np.ndarray:
    # todo(amylase): implement erfc in _erf.py and use it for big |a| inputs.
    return 0.5 + 0.5 * erf(a / 2**0.5)


@functools.lru_cache(1000)
def _log_ndtr_single(a: float) -> float:
    if a > 6:
        return -_ndtr_single(-a)
    if a > -20:
        return math.log(_ndtr_single(a))

    log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
    denom_cons = 1 / a**2
    sign = 1
    i = 0

    while abs(last_total - right_hand_side) > sys.float_info.epsilon:
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        right_hand_side += sign * numerator * denom_factor

    return log_LHS + math.log(right_hand_side)


def _log_ndtr(a: np.ndarray) -> np.ndarray:
    return np.frompyfunc(_log_ndtr_single, 1, 1)(a).astype(float)


def _norm_logpdf(x: np.ndarray) -> np.ndarray:
    return -(x**2) / 2.0 - _norm_pdf_logC


def _log_gauss_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Log of Gaussian probability mass within an interval"""

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _log_diff(_log_ndtr(b), _log_ndtr(a))

    def mass_case_right(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return mass_case_left(-b, -a)

    def mass_case_central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return np.log1p(-_ndtr(a) - _ndtr(-b))

    # _lazyselect not working; don't care to debug it
    out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
    if a[case_left].size:
        out[case_left] = mass_case_left(a[case_left], b[case_left])
    if a[case_right].size:
        out[case_right] = mass_case_right(a[case_right], b[case_right])
    if a[case_central].size:
        out[case_central] = mass_case_central(a[case_central], b[case_central])
    return np.real(out)  # discard ~0j


def _bisect(f: Callable[[float], float], a: float, b: float, c: float) -> float:
    if f(a) > c:
        a, b = b, a
    # In the algorithm, it is assumed that all of (a + b), (a * 2), and (b * 2) are finite.
    for _ in range(100):
        m = (a + b) / 2
        if a == m or b == m:
            return m
        if f(m) < c:
            a = m
        else:
            b = m
    return (a + b) / 2


def _ndtri_exp_single(y: float) -> float:
    # TODO(amylase): Justify this constant
    return _bisect(_log_ndtr_single, -100, +100, y)


def _ndtri_exp(y: np.ndarray) -> np.ndarray:
    return np.frompyfunc(_ndtri_exp_single, 1, 1)(y).astype(float)


def ppf(q: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    q, a, b = np.atleast_1d(q, a, b)
    q, a, b = np.broadcast_arrays(q, a, b)

    case_left = a < 0
    case_right = ~case_left

    def ppf_left(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr(a), np.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)

    def ppf_right(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr(-b), np.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)

    out = np.empty_like(q)

    q_left = q[case_left]
    q_right = q[case_right]

    if q_left.size:
        out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
    if q_right.size:
        out[case_right] = ppf_right(q_right, a[case_right], b[case_right])

    out[q == 0] = a[q == 0]
    out[q == 1] = b[q == 1]
    out[a == b] = math.nan

    return out


def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    random_state = random_state or np.random.RandomState()
    size = np.broadcast(a, b, loc, scale).shape
    percentiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(percentiles, a, b) * scale + loc


def logpdf(
    x: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
) -> np.ndarray:
    x = (x - loc) / scale

    x, a, b = np.atleast_1d(x, a, b)

    out = _norm_logpdf(x) - _log_gauss_mass(a, b) - np.log(scale)

    x, a, b = np.broadcast_arrays(x, a, b)
    out[(x < a) | (b < x)] = -np.inf
    out[a == b] = math.nan

    return out
