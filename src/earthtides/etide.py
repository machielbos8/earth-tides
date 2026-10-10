#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# etide: solid Earth tide displacement at a station, from the Tamura (1987)
# tidal potential.
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
"""etide: solid Earth tide displacement at a station, from the Tamura (1987) tidal potential.

A small program (NumPy only; it does not need the solver of earth-tides). The displacement is a harmonic synthesis over the
1200 waves of Tamura (1987), degrees 2 to 4,

    u_r = sum_k A_k Re[h_k e^{i chi_k}] W_k,        u_N = sum_k A_k Re[l_k e^{i chi_k}] dW_k/dphi,
    u_E = sum_k A_k Re[i m_k l_k e^{i chi_k}] W_k / cos(phi),

with W the potential divided by gravity (normalized at the equatorial radius, as in the IERS
Conventions), phi geocentric latitude and chi_k the astronomical argument; plus the l(1) terms
for degree 2. The Love numbers come from a table per band and frequency (see LoveModel):

  'iers'  IERS Conventions 2010 (TN36): the resonance formula for the diurnal waves and the
          anelastic formula for the zonal waves, evaluated at the frequency of every wave, step-1
          values for the semidiurnal waves; degree 3: h3 = 0.292, l3 = 0.015; degree 4 from PREM.
  'ddw'   as 'iers', with the diurnal h(0), l(0) of Dehant, Defraigne & Wahr (1999, Table 14a).
  'prem'  anelastic, hydrostatic PREM of earth-tides, one Love number per wave group (default
          groups, < 0.03 mm from per wave; data/love_PREM.json); 'prem-allwaves': per wave
          (data/love_PREM_allwaves.json).
  file    any Love-number table written by earthtides-love (JSON, IERS conventions only).

The astronomical angles and the normalized spherical functions follow Tamura (1987) and the
gravity-tide program of geoido (M. Bos), from which this program is derived.

Usage:
  etide LAT LON [H] --start 2026-01-01 --days 30 --step 3600 [--love iers|mhb|ddw|prem|FILE]
  -> columns: MJD (UTC), radial (up), north, east displacement [m]

Tamura, Y. (1987). A harmonic development of the tide-generating potential. Bull. Inf. Marees
  Terrestres 99, 6813-6855.
Dehant, V., Defraigne, P. & Wahr, J. M. (1999). Tides for a convective Earth. JGR 104, 1035-1058.
Petit, G. & Luzum, B. (eds) (2010). IERS Conventions (2010). IERS Technical Note 36.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RAD = np.pi / 180.0
GE = 3.986004418e14              # GM of the Earth [m^3/s^2]
RE = 6378136.6                   # equatorial radius [m] (IERS)
F_ELL = 1.0 / 298.257222101      # flattening of the reference ellipsoid (GRS80)
# Tamura's scale of the potential: 3/4 mu GE sin^3(pi_moon) / Re^2 (as in geoido)
MU, SINPI, RE_T = 0.012300034, 3422.448, 6378137.0
CV = 0.75 * MU * GE * (SINPI / 3600.0 * RAD)**3 / RE_T**2
# TAI - UTC: MJD of each change and the new value [s]
LEAP = np.array([[41317, 10], [41499, 11], [41683, 12], [42048, 13], [42413, 14], [42778, 15],
                 [43144, 16], [43509, 17], [43874, 18], [44239, 19], [44786, 20], [45151, 21],
                 [45516, 22], [46247, 23], [47161, 24], [47892, 25], [48257, 26], [48804, 27],
                 [49169, 28], [49534, 29], [50083, 30], [50630, 31], [51179, 32], [53736, 33],
                 [54832, 34], [56109, 35], [57204, 36], [57754, 37]], float)


# ---------------------------------------------------------------------------- catalogue
class Tamura:
    """The 1200 waves: degree n, Doodson multipliers (8), phase (units of 90 deg),
    amplitude A and its rate dA per century."""

    def __init__(self, fname=os.path.join(DATA, 'Tamura.dat')):
        a = np.loadtxt(fname)
        self.n = a[:, 0].astype(int)
        self.dood = a[:, 1:9].astype(int)
        self.m = self.dood[:, 0]
        self.phase = a[:, 9]
        self.A, self.dA = a[:, 10], a[:, 11]
        # speeds of the 8 angles [deg / Julian century] -> frequency [cycles per day]
        speed = np.array([12703732.89, 481267.881342, 36000.769822, 4069.013711, 1934.136185,
                          1.719533, 32964.47, 22518.44])
        self.f_cpd = self.dood @ speed / (360.0 * 36525.0)


def tai_utc(mjd):
    i = np.searchsorted(LEAP[:, 0], mjd, side='right') - 1
    if np.any(i < 0):
        raise ValueError('date before 1972')
    return LEAP[i, 1]


def angles(mjd, lon):
    """Tamura's eight astronomical angles [deg], shape (8, nt), at UTC MJD and east longitude
    lon [deg]; and dynamical time Td [Julian centuries since J2000]."""
    mjd = np.atleast_1d(np.asarray(mjd, float))
    Tu = (mjd - 51544.5) / 36525.0
    Tt = (mjd + (tai_utc(mjd) + 32.184) / 86400.0 - 51544.5) / 36525.0
    g = RAD * (357.53 + 0.9856003 * (mjd - 51544.5))
    Td = Tt + (0.001658 * np.sin(g) + 0.000014 * np.sin(2 * g)) / 3.15576e9
    t = (mjd - np.floor(mjd)) * 24.0
    am = 280.4606184 + 36000.7700536 * Tu + 0.00038793 * Tu**2 - 0.0000000258 * Tu**3
    s = 218.316656 + 481267.881342 * Td - 0.001330 * Td**2
    h = 280.466449 + 36000.769822 * Td + 0.0003036 * Td**2
    f = np.empty((8, len(mjd)))
    f[0] = 15.0 * t + am - s + lon
    f[1] = s + 0.0040 * np.cos(RAD * (29.0 + 133.0 * Td))
    f[2] = h + 0.0018 * np.cos(RAD * (159.0 + 19.0 * Td))
    f[3] = 83.353243 + 4069.013711 * Td - 0.010324 * Td**2
    f[4] = 234.955444 + 1934.136185 * Td - 0.002076 * Td**2
    f[5] = 282.937348 + 1.719533 * Td + 0.0004597 * Td**2
    f[6] = 248.1 + 32964.47 * Td
    f[7] = 81.5 + 22518.44 * Td
    return f, Td


# ---------------------------------------------------------------------------- geometry
_GAMMA = {(2, 0): 2.0, (2, 1): 1.0, (2, 2): 1.0, (3, 0): 2 / np.sqrt(5), (3, 1): 16 / (3 * np.sqrt(15)),
          (3, 2): 2 / (3 * np.sqrt(3)), (3, 3): 1.0, (4, 0): 8.0,
          (4, 1): (3 + np.sqrt(393)) * np.sqrt(390 + 2 * np.sqrt(393)) / 224, (4, 2): 9 / 7,
          (4, 3): 3 * np.sqrt(3) / 16, (4, 4): 1.0}


def F_nm(n, m, phi):
    """Tamura's (Doodson-normalized) spherical functions of geocentric latitude phi [rad]."""
    s, c = np.sin(phi), np.cos(phi)
    P = {(2, 0): 1 - 3 * s**2, (2, 1): 2 * s * c, (2, 2): c**2,
         (3, 0): s * (3 - 5 * s**2), (3, 1): c * (1 - 5 * s**2), (3, 2): s * c**2, (3, 3): c**3,
         (4, 0): 3 - 30 * s**2 + 35 * s**4, (4, 1): 2 * s * c * (3 - 7 * s**2),
         (4, 2): c**2 * (1 - 7 * s**2), (4, 3): s * c**3, (4, 4): c**4}[(n, m)]
    return P / _GAMMA[(n, m)]


def geocentric(lat, h=0.0):
    """Geocentric latitude [rad] and radius [m] of a point at geodetic latitude lat [deg]."""
    la = lat * RAD
    e2 = F_ELL * (2 - F_ELL)
    N = 6378137.0 / np.sqrt(1 - e2 * np.sin(la)**2)
    x, z = (N + h) * np.cos(la), (N * (1 - e2) + h) * np.sin(la)
    return np.arctan2(z, x), np.hypot(x, z)


def W_functions(n, m, phi, r=RE):
    """Potential / gravity per unit wave amplitude and its phi-derivative, at radius r."""
    g = GE / RE**2
    k = CV * RE * (r / RE)**n / g
    d = 1e-6
    return k * F_nm(n, m, phi), k * (F_nm(n, m, phi + d) - F_nm(n, m, phi - d)) / (2 * d)


# ---------------------------------------------------------------------------- Love numbers
# IERS Conventions 2010 (TN36), Table 7.1 and eq. 6.10: diurnal resonance formula
# L(sigma) = L0 + sum_a L_a / (sigma - sigma_a), sigma in cycles per sidereal day (CW, FCN, FICN)
_MHB_SIGMA = np.array([-0.0026010 - 0.0001361j, 1.0023181 + 0.000025j, 0.999026 + 0.000780j])
_MHB = dict(h0=[0.60671 - 0.2420e-2j, -0.15777e-2 - 0.7630e-4j, 0.18053e-3 - 0.6292e-5j,
                -0.18616e-5 + 0.1379e-6j],
            h2=[-0.615e-3 - 0.122e-4j, 0.160e-5 + 0.116e-6j, 0.201e-6 + 0.279e-8j,
                -0.329e-7 - 0.217e-8j],
            l0=[0.84963e-1 - 0.7395e-3j, -0.22107e-3 - 0.9646e-5j, -0.54710e-5 - 0.2990e-6j,
                -0.29904e-7 - 0.7717e-8j],
            l1=[0.121e-2 + 0.136e-6j, -0.316e-5 - 0.166e-6j, 0.272e-6 - 0.858e-8j,
                -0.545e-8 + 0.827e-11j],
            l2=[0.19334e-3 - 0.3819e-5j, -0.50331e-6 - 0.1639e-7j, -0.66460e-8 + 0.5076e-9j,
                0.10372e-7 + 0.7511e-9j])


# TN36 Table 7.2, diurnal: (frequency [cpsd], h(0)R, h(0)I) and (frequency, l(0)R, l(0)I). These
# include the resonant ocean-tide loading correction and the deviation from the resonance formula.
_T72_H = np.array([[0.85461, .6039, -.0027], [0.85946, .6039, -.0026], [0.89066, .6036, -.0026],
                   [0.89080, .6036, -.0026], [0.89565, .6035, -.0026], [0.92685, .6028, -.0025],
                   [0.92700, .6028, -.0025], [0.93246, .6026, -.0025], [0.95835, .6011, -.0024],
                   [0.96381, .6005, -.0023], [0.96865, .5998, -.0023], [0.99181, .5878, -.0015],
                   [0.99454, .5817, -.0011], [0.99727, .5692, -.0004], [0.99985, .5283, .0023],
                   [1.00000, .5236, .0030], [1.00015, .5182, .0036], [1.00029, .5120, .0043],
                   [1.00273, 1.0569, .0036], [1.00288, .9387, -.0050], [1.00546, .6645, -.0059],
                   [1.03135, .6117, -.0030], [1.03619, .6108, -.0030], [1.07300, .6080, -.0028]])
_T72_L = np.array([[0.89080, .0846, -.0006], [0.92685, .0846, -.0006], [0.92700, .0846, -.0006],
                   [0.96381, .0847, -.0006], [0.99454, .0853, -.0006], [0.99985, .0869, -.0006],
                   [1.00000, .0870, -.0006], [1.00015, .0872, -.0006], [1.00273, .0710, -.0020],
                   [1.00546, .0828, -.0007], [1.03619, .0845, -.0006], [1.07300, .0846, -.0006]])


def table72(key, f_cpd):
    """TN36 diurnal h(0) or l(0): the resonance formula plus the correction (Table 7.2 minus formula)
    interpolated linearly in frequency between the tabulated lines."""
    T = _T72_H if key == 'h0' else _T72_L
    sig_t = T[:, 0]
    d = T[:, 1] + 1j * T[:, 2] - mhb_resonance(key, sig_t * 1.002737909)
    sig = np.asarray(f_cpd) / 1.002737909
    corr = np.interp(sig, sig_t, d.real) + 1j * np.interp(sig, sig_t, d.imag)
    return mhb_resonance(key, f_cpd) + corr


def mhb_resonance(key, f_cpd):
    sigma = np.asarray(f_cpd) / 1.002737909
    c = _MHB[key]
    return c[0] + sum(c[a + 1] / (sigma - _MHB_SIGMA[a]) for a in range(3))


def zonal_anelastic(key, f_cpd):
    """TN36 eqs. 7.4a,b: zonal h(0), l(0) with mantle anelasticity (f_m = 1/200 s, alpha = 0.15)."""
    L0, dL = dict(h0=(0.5998, 9.96e-4), l0=(0.0831, 3.01e-4))[key]
    alpha, fm = 0.15, 86400.0 / 200.0                 # f_m in cycles per day
    x = (fm / np.asarray(f_cpd))**alpha
    return L0 - dL * (1 / np.tan(alpha * np.pi / 2) * (1 - x) + 1j * x)


# DDW (1999) Table 14a, non-hydrostatic inelastic Earth: L(x) = b0 + (x - x_O1) sum_a a/(x_a - x),
# x = -frequency in cycles per sidereal day (real parts only)
_DDW_X = dict(O1=-0.9269978, CW=0.0024854, FCN=-1.0023182, P=-26.9671960, M=26.1731065)
_DDW = dict(h0=(0.604589, -0.0016357, -0.0026229, 0.29238, 0.28213),
            l0=(0.0855189, -0.00022646, 0.000088571, 0.0054848, 0.0071628))


def ddw_resonance(key, f_cpd):
    x = -np.asarray(f_cpd) / 1.0027379
    b0, acw, afcn, ap, am = _DDW[key]
    X = _DDW_X
    return b0 + (x - X['O1']) * (acw / (X['CW'] - x) + afcn / (X['FCN'] - x)
                                 + ap / (X['P'] - x) + am / (X['M'] - x))


class LoveModel:
    """Love numbers per wave: arrays h0, h2, l0, l2, l1 (complex) for the degree-2 waves and
    h0, l0 for degrees 3 and 4.

    'iers': TN36. Semidiurnal: step-1 values h(0) = 0.6078 - 0.0022i, l(0) = 0.0847 - 0.0007i,
            h(2) = -0.0006, l(2) = 0.0002, l(1) = 0.0024. Diurnal: the resonance formula with the
            coefficients of Table 7.1 at the frequency of every wave. Zonal: eqs. 7.4a,b at the
            frequency of every wave (the permanent tide with h(0) = 0.6078, l(0) = 0.0847).
            Degree 3: h = 0.292, l = 0.015. Degree 4 (not in TN36): h = 0.179, l = 0.011, the
            spherical, anelastic PREM values of earthtides.
            The diurnal h(0), l(0) include the resonant ocean-loading correction of Table 7.2.
    'mhb':  as 'iers', diurnal values from the resonance formula alone (Table 7.1).
    'ddw':  as 'mhb', but the real parts of the diurnal h(0), l(0) from DDW (1999).
    'prem': the table data/love_PREM.json (earthtides-love, default wave groups);
    'prem-allwaves': data/love_PREM_allwaves.json (earthtides-love --all-waves).
    file:   a JSON table written by earthtides-love, in the IERS conventions, covering every wave
            of Tamura.dat (matched by degree and Doodson multipliers).
    """

    def __init__(self, cat, source='iers'):
        n, m, f = cat.n, cat.m, cat.f_cpd
        z = np.zeros(len(n), complex)
        self.h0, self.h2, self.l0, self.l2, self.l1 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
        if source in ('iers', 'mhb', 'ddw'):
            d2 = n == 2
            sd, di, lp = d2 & (m == 2), d2 & (m == 1), d2 & (m == 0)
            self.h0[sd], self.l0[sd] = 0.6078 - 0.0022j, 0.0847 - 0.0007j
            self.h2[d2], self.l2[d2], self.l1[sd] = -0.0006, 0.0002, 0.0024
            for key in ('h0', 'h2', 'l0', 'l1', 'l2'):
                getattr(self, key)[di] = mhb_resonance(key, f[di])
            if source == 'iers':
                for key in ('h0', 'l0'):
                    getattr(self, key)[di] = table72(key, f[di])
            if source == 'ddw':
                for key in ('h0', 'l0'):
                    v = getattr(self, key)
                    v[di] = ddw_resonance(key, f[di]) + 1j * v[di].imag
            zero = lp & (np.abs(f) < 1e-6)
            for key in ('h0', 'l0'):
                getattr(self, key)[lp & ~zero] = zonal_anelastic(key, f[lp & ~zero])
            self.h0[zero], self.l0[zero] = 0.6078, 0.0847
            self.h0[n == 3], self.l0[n == 3] = 0.292, 0.015
            self.h0[n == 4], self.l0[n == 4] = 0.179, 0.011   # not in TN36: anelastic PREM
        else:
            path = {'prem': 'love_PREM.json', 'prem-allwaves': 'love_PREM_allwaves.json'}.get(source)
            self._read_json(cat, os.path.join(DATA, path) if path else source)

    def _read_json(self, cat, path):
        with open(path) as fp:
            t = json.load(fp)
        if t.get('format') != 'earthtides-love':
            raise ValueError(f'{path}: not a Love-number table of earthtides-love')
        if not t['metadata'].get('iers_conventions'):
            raise ValueError(f'{path}: the table is not in the IERS conventions '
                             f'({t["metadata"].get("conventions")}); etide cannot use it')
        table = {(e['n'], tuple(e['doodson'])): e for e in t['entries'] if 'doodson' in e}
        missing = 0
        for i, (n, d) in enumerate(zip(cat.n, cat.dood)):
            e = table.get((int(n), tuple(int(x) for x in d)))
            if e is None:
                missing += 1
                continue
            for key in ('h0', 'h2', 'l0', 'l2', 'l1'):
                if key in e:
                    getattr(self, key)[i] = complex(*e[key])
        if missing:
            raise ValueError(f'{path}: {missing} waves of the Tamura catalogue are not in the table')
        self.metadata = t['metadata']


# ---------------------------------------------------------------------------- synthesis
def displacement(mjd, lat, lon, h=0.0, love='iers', degrees=(2, 3, 4), cat=None):
    """Radial (up), north and east displacement [m] at UTC MJD (array) for a station at geodetic
    latitude lat, east longitude lon [deg] and height h [m]; components along the geocentric
    radius and its perpendiculars."""
    cat = cat or Tamura()
    L = love if isinstance(love, LoveModel) else LoveModel(cat, love)
    phi, _ = geocentric(lat, h)
    f, Td = angles(mjd, lon)
    chi = RAD * (cat.dood @ f + 90.0 * cat.phase[:, None])            # (nw, nt)
    amp = cat.A[:, None] + Td[None, :] * cat.dA[:, None]
    E = amp * np.exp(1j * chi)
    sp, cp = np.sin(phi), np.cos(phi)
    P2 = 1.5 * sp**2 - 0.5
    ur = np.zeros(E.shape[1])
    un, ue = ur.copy(), ur.copy()
    for n in degrees:
        for m in range(n + 1):
            s = (cat.n == n) & (cat.m == m)
            if not s.any():
                continue
            W, dW = W_functions(n, m, phi)
            hh = L.h0[s] + L.h2[s] * P2
            ll = L.l0[s] + L.l2[s] * P2
            ur += np.real((hh[:, None] * E[s]).sum(0)) * W
            un += np.real((ll[:, None] * E[s]).sum(0)) * dW
            ue += np.real((1j * m * ll[:, None] * E[s]).sum(0)) * W / cp
            if n == 2 and m > 0:                 # l(1) terms (TN36 eqs. 7.12, 7.13)
                kappa = {1: 2 / 3, 2: 1 / 3}[m]      # F_2m = kappa * (3 sin phi cos phi, 3 cos^2 phi)
                Wn = CV * RE / (GE / RE**2) * kappa   # potential / gravity per unit of the TN36 function
                #   (= W / F_nm(2, m, phi) * kappa, without dividing by F_21 = 0 at the equator)
                N1 = {1: -3 * sp**2, 2: -6 * sp * cp}[m]
                E1 = {1: -3j * sp * np.cos(2 * phi), 2: 6j * cp * sp**2}[m]
                l1E = (L.l1[s][:, None] * E[s]).sum(0)
                un += np.real(l1E * N1) * Wn
                ue += np.real(l1E * E1) * Wn
    return ur, un, ue


def main():
    p = argparse.ArgumentParser(
        description='Body-tide displacement of a station (Tamura 1987 potential, degrees 2-4), '
                    'including the permanent tide. Output: MJD (UTC), up, north, east [m]; up along '
                    'the geocentric radius.')
    p.add_argument('lat', type=float, help='geodetic latitude [deg]')
    p.add_argument('lon', type=float, help='east longitude [deg]')
    p.add_argument('h', type=float, nargs='?', default=0.0,
                   help='ellipsoidal height [m] (default 0; enters only the geocentric latitude)')
    p.add_argument('--start', default='2026-01-01', help='UTC start, YYYY-MM-DD[THH:MM] (default 2026-01-01)')
    p.add_argument('--days', type=float, default=1.0, help='length [days] (default 1)')
    p.add_argument('--step', type=float, default=3600.0, help='sampling [s] (default 3600)')
    p.add_argument('--love', default='iers',
                   help="iers (default), mhb, ddw, prem, prem-allwaves, or a JSON table of "
                        "earthtides-love")
    a = p.parse_args()
    t0 = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    mjd0 = (t0 - datetime(1858, 11, 17, tzinfo=timezone.utc)).total_seconds() / 86400.0
    mjd = mjd0 + np.arange(0, a.days * 86400.0, a.step) / 86400.0
    ur, un, ue = displacement(mjd, a.lat, a.lon, a.h, a.love)
    print('#     MJD(UTC)      up [m]      north [m]     east [m]')
    for row in zip(mjd, ur, un, ue):
        print(f'{row[0]:14.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}')


if __name__ == '__main__':
    main()
