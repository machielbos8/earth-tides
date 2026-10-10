# -*- coding: utf-8 -*-
#
# From Love numbers to station displacements: harmonic synthesis with the
# HARDISP tidal catalog.
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
"""From Love numbers to station displacements: harmonic synthesis with the HARDISP tidal catalog.

HARDISP (IERS Conventions software, D. Agnew) writes a displacement as a sum over 342 degree-2
tidal lines k,

    u(t) = sum_k TAMP_k Re[ Z_k exp(i (arg_k(t) + off_m)) ],

with TAMP_k the Cartwright-Tayler-Edden amplitude [m], arg_k the Doodson argument (TDFRPH),
off_m = 180, 90, 0 degrees for the long-period (m = 0), diurnal (m = 1) and semidiurnal (m = 2)
bands, and Z_k the complex admittance. HARDISP takes Z at 11 constituents from a BLQ file and
interpolates it with a spline in frequency (ADMINT). Here Z can be given for every line, so the
spline can be compared with the exact admittance (the free core nutation resonance in the
diurnal band).

For the body tide at a station at colatitude theta, east longitude lam, the admittance of a line
of order m and frequency f is

    Z_r = h(f) c_m(theta) e^{i m lam},
    Z_s = l(f) dc_m/dtheta e^{i m lam},                     (south = +theta)
    Z_w = -i m l(f) c_m(theta) / sin(theta) e^{i m lam},    (west = -lambda)

where c_m are the Cartwright-Tayler functions in which TAMP is the amplitude of V/g (the
function `ct`); h and l are the station Love numbers (they may depend on the station).
A body-tide BLQ for HARDISP holds |Z| |TAMP| and the phase lag -arg(Z) of the 11 constituents.
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
# The 11 HARDISP/BLQ constituents: M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa
BLQ_NAMES = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa']
BLQ_DOODSON = np.array([[2, 0, 0, 0, 0, 0], [2, 2, -2, 0, 0, 0], [2, -1, 0, 1, 0, 0],
                        [2, 2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0],
                        [1, 1, -2, 0, 0, 0], [1, -2, 0, 1, 0, 0], [0, 2, 0, 0, 0, 0],
                        [0, 1, 0, -1, 0, 0], [0, 0, 2, 0, 0, 0]])


def catalog():
    """Doodson multipliers (342, 6) and amplitudes TAMP (342,) of the HARDISP catalog."""
    a = np.loadtxt(DATA / 'hardisp_catalog.txt')
    return a[:, :6].astype(int), a[:, 6]


def _fundamental(T):
    """Delaunay-type angles F1..F5 of TDFRPH [deg], T in Julian centuries TT since J2000."""
    poly = lambda c: c[0] + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))
    F1 = poly([134.9634025100, 477198.8675605000, 0.0088553333, 0.0000143431, -0.0000000680])
    F2 = poly([357.5291091806, 35999.0502911389, -0.0001536667, 0.0000000378, -0.0000000032])
    F3 = poly([93.2720906200, 483202.0174577222, -0.0035420000, -0.0000002881, 0.0000000012])
    F4 = poly([297.8501954694, 445267.1114469445, -0.0017696111, 0.0000018314, -0.0000000088])
    F5 = poly([125.0445550100, -1934.1362619722, 0.0020756111, 0.0000021394, -0.0000000165])
    return F1, F2, F3, F4, F5


def doodson_angles(jd_utc, tt_utc=69.184):
    """The six Doodson angles of TDFRPH [deg], shape (6, nt), for UTC Julian dates."""
    jd_utc = np.atleast_1d(np.asarray(jd_utc, float))
    dayfr = np.mod(jd_utc + 0.5, 1.0)
    T = (jd_utc - 2451545.0 + tt_utc / 86400.0) / 36525.0
    F1, F2, F3, F4, F5 = _fundamental(T)
    D = np.empty((6, len(jd_utc)))
    D[0] = 360.0 * dayfr - F4
    D[1] = F3 + F5
    D[2] = D[1] - F4
    D[3] = D[1] - F1
    D[4] = -F5
    D[5] = D[2] - F2
    return D


def doodson_freq(idd, T=0.26):
    """Frequencies [cycles per day] of lines with Doodson multipliers idd (n, 6), as TDFRPH."""
    FD1 = 0.0362916471 + 0.0000000013 * T
    FD2 = 0.0027377786
    FD3 = 0.0367481951 - 0.0000000005 * T
    FD4 = 0.0338631920 - 0.0000000003 * T
    FD5 = -0.0001470938 + 0.0000000003 * T
    DD = np.array([1.0 - FD4, FD3 + FD5, FD3 + FD5 - FD4, FD3 + FD5 - FD1, -FD5,
                   FD3 + FD5 - FD4 - FD2])
    return np.atleast_2d(idd) @ DD


def ct(m, theta, deriv=False):
    """Cartwright-Tayler degree-2 functions c_2^m(theta) (TAMP is the amplitude of V/g in them),
    or their theta derivative."""
    c, s = np.cos(theta), np.sin(theta)
    if m == 0:      # sign: with HARDISP's 180-degree offset of the long-period band (checked
        k = -np.sqrt(5 / (4 * np.pi))       # against DEHANTTIDEINEL)
        return k * (-3 * c * s) if deriv else k * (1.5 * c**2 - 0.5)
    if m == 1:
        k = 3 * np.sqrt(5 / (24 * np.pi))
        return k * (c**2 - s**2) if deriv else k * s * c
    k = 3 * np.sqrt(5 / (96 * np.pi))
    return k * 2 * s * c if deriv else k * s**2


def synthesize(jd_utc, Z, tt_utc=69.184, lines=None):
    """u(t) = sum_k TAMP_k Re[Z_k exp(i(arg_k + off_m))] for complex admittances Z (342,) (or
    (342, ncomp)); `lines` restricts the sum (boolean mask)."""
    idd, tamp = catalog()
    D = doodson_angles(jd_utc, tt_utc)
    arg = np.deg2rad(idd @ D + np.where(idd[:, 0] == 0, 180.0, np.where(idd[:, 0] == 1, 90.0, 0.0))[:, None])
    if lines is not None:
        arg, tamp, Z = arg[lines], tamp[lines], Z[lines]
    E = np.exp(1j * arg)                                          # (nl, nt)
    Z = np.asarray(Z)
    if Z.ndim == 1:
        return (tamp * Z) @ E
    return np.real(np.einsum('k,kc,kt->tc', tamp, Z, E))


def body_admittance(theta, lam, m, h, l_n, l_e=None):
    """Admittances (Z_r, Z_s, Z_w) of a degree-2 line of order m at a station (colatitude theta,
    east longitude lam [rad]) for station Love numbers h (radial), l_n (north-south) and l_e
    (east-west; = l_n on a sphere)."""
    l_e = l_n if l_e is None else l_e
    e = np.exp(1j * m * lam)
    Zr = h * ct(m, theta) * e
    Zs = l_n * ct(m, theta, deriv=True) * e
    Zw = -1j * m * l_e * ct(m, theta) / np.sin(theta) * e
    return Zr, Zs, Zw


def blq(Zr, Zs, Zw):
    """BLQ text (amplitudes [m], phase lags [deg]) from admittances at the 11 BLQ constituents
    (arrays of shape (11,)), in the order radial, west, south of the BLQ format."""
    idd, tamp = catalog()
    ta = np.array([abs(tamp[np.where((idd == d).all(1))[0][0]]) for d in BLQ_DOODSON])
    rows = []
    for Z in (Zr, Zw, Zs):
        rows.append(' ' + ''.join(f'{v:7.5f}' for v in np.abs(Z) * ta))
    for Z in (Zr, Zw, Zs):
        rows.append(' ' + ''.join(f'{v:7.1f}' for v in -np.degrees(np.angle(Z))))
    return '\n'.join(rows) + '\n'
