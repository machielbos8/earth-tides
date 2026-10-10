# -*- coding: utf-8 -*-
#
# Basis functions of the solver: Chebyshev polynomials in the radial direction
# and Schmidt semi-normalized associated Legendre functions in colatitude,
# with their derivatives.
#
# This file is part of earth-tides 1.0.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 8/10/2026 Machiel Bos
#===============================================================================
"""Basis functions of the solver: Chebyshev polynomials in the radial direction and Schmidt
semi-normalized associated Legendre functions in colatitude, with their derivatives.

cheb_basis(x, N)  values, first and second derivatives of T_0 ... T_{N-1} at x in [-1, 1];
                  the derivatives are again Chebyshev series (numpy.polynomial.chebyshev.chebder)
ybar(n, m, theta) Ybar_n^m(cos theta) and its theta-derivatives, computed exactly from the
                  trigonometric polynomial sin^m(theta) Q(cos theta) by an FFT
"""
import numpy as np
from numpy.polynomial import chebyshev as C


def cheb_basis(x, N):
    """T_k(x), T_k'(x), T_k''(x) for k < N; arrays (N, len(x))."""
    x = np.atleast_1d(x)
    eye = np.eye(N)
    T = C.chebval(x, eye)
    T1 = C.chebval(x, C.chebder(eye, 1)) if N > 1 else 0 * T
    T2 = C.chebval(x, C.chebder(eye, 2)) if N > 2 else 0 * T
    return T, T1, T2


def _trig_derivs(fun, deg, theta, nder):
    """Values and theta-derivatives of a trigonometric polynomial."""
    M = 4 * (deg + 4)
    tg = 2 * np.pi * np.arange(M) / M
    c = np.fft.fft(fun(tg)) / M
    k = np.fft.fftfreq(M, 1.0 / M)
    E = np.exp(1j * np.multiply.outer(np.asarray(theta, float), k))
    return [np.real(E @ (c * (1j * k) ** p)) for p in range(nder + 1)]


def ybar(n, m, theta, nder=3, over_sin=False):
    """Schmidt P_n^m(cos theta) (or P_n^m / sin theta) and theta derivatives."""
    p = m - 1 if over_sin else m
    fun = lambda t: np.sin(t) ** p * _schmidt_reduced(n, m, np.cos(t))
    return _trig_derivs(fun, n, theta, nder)


def _schmidt_reduced(n, m, x):
    """Schmidt-normalized P_n^m(x) / sin^m, by the stable three-term recursion
    (no Condon-Shortley phase), so that degrees of several hundred are exact."""
    cm = 1.0
    for k in range(2, m + 1):
        cm *= np.sqrt((2 * k - 1) / (2 * k))
    prev, cur = 0.0 * x, cm + 0.0 * x                     # n = m - 1, n = m
    for k in range(m + 1, n + 1):
        prev, cur = cur, ((2 * k - 1) * x * cur - np.sqrt((k - 1)**2 - m**2) * prev) / np.sqrt(k**2 - m**2)
    return cur


def legendre_theta(n, theta, m=0):
    """Ybar_n^m and its first three theta derivatives."""
    return tuple(ybar(n, m, theta, 3))
