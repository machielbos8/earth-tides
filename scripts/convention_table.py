# -*- coding: utf-8 -*-
#
# Paper 1: sensitivity of the IERS-form Love numbers to the conventions behind
# them.
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
# 7/10/2026 Machiel Bos
#===============================================================================
"""Paper 1: sensitivity of the IERS-form Love numbers to the conventions behind them.

One PREM solution per tide (O1: m = 1, M2: m = 2; fluid core, Coriolis, inertia, consistent reference
state). Baseline = this paper / IERS TN36: geocentric latitude in the basis functions, displacement at
the deformed (flattened) surface, components along the geocentric radius, potential normalised at the
equatorial radius R_e, mantle (Tisserand) frame for m = 1. Each row changes ONE convention:
  frame      Wahr's surface frame (no net rigid rotation of the surface) instead of the mantle frame
  geodetic   basis functions at the geodetic latitude phi + f sin 2 phi
  normal     components along the ellipsoid normal: u_n = u_r - d u_th, u_th' = u_th - d u_r,
             d = f sin 2 phi (angle between normal and radius)
  normal+geo normal components and geodetic latitude together (a geodetic system)
  mean sph   displacement at the mean sphere r = a instead of the flattened surface (first order:
             u(a) - delta/R_s d_s u, delta = -2/3 f a P2; rigid rotation removed before the shift,
             d_r of a rigid rotation = value / r)
  norm a     potential normalised at the mean radius a instead of R_e (all numbers x (1+f/3)^4)
  no centr.  (M2 only) reference gravity without the centrifugal term (Greff-Lefftz et al. 2005);
             for m = 1 the force-free tilt needs the consistent reference, so not done there
Usage: python scripts/convention_table.py  ->  results/convention_table.txt
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, ybar, cheb_basis
from earthtides.iers import iers_fit_fields

md = EarthModel('data/PREM.dat')
geo = ClairautGeometry(md)
TIDES = {'O1': (1, 0.92700), 'M2': (2, 2 * 0.9661368)}
th = np.linspace(0.03, np.pi - 0.03, 600)
a = md.a
iS = md.n_layers - 1
e, de, _ = geo.eps(iS, np.array([a]))
f, df = e[0], de[0]
P2c = 0.5 * (3 * np.cos(th)**2 - 1)
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def solve(m, fr, frame='mantle', centrifugal=True):
    S = TidalSolver(md, geo, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=fr * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH if centrifugal else 0.0, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=centrifugal, frame=frame)
    S.solve()
    return S


def raw_fields(S, deriv=False):
    """u_r, u_theta, u_lambda at s = a (or their d/ds), no rigid-rotation removal."""
    N, nb = S.N, S.nfld * S.N
    Tp, T1, _ = cheb_basis(np.array([1.0]), N)
    sc = 2.0 / (md.r1[iS] - md.r0[iS])
    v = {}
    for j, fld in enumerate(S.fields):
        c = S.coef[iS * nb + j * N: iS * nb + (j + 1) * N]
        v[fld] = float(c @ T1[:, 0] * sc) if deriv else float(c @ Tp[:, 0])
    m = S.m
    ur = sum(v[('U', n)] * ybar(n, m, th, 0)[0] for n in S.deg)
    ut, up = 0 * th, 0 * th
    for n in S.degV:
        y1 = ybar(n, m, th, 1)[1]
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        ut, up = ut + v[('V', n)] * y1, up - v[('V', n)] * ys
    for n in S.degW:
        y1 = ybar(n, m, th, 1)[1]
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        ut, up = ut - v[('W', n)] * ys, up + v[('W', n)] * y1
    return ur, ut, up


def body(S, ur, ut, up):
    if S.m == 1:
        ut, up = (S._remove_mantle_rotation(th, ut, up) if S.frame == 'mantle'
                  else S._remove_rigid_rotation(th, ut, up))
    return ur, ut, up


norm_e = (a * (1 + f / 3.0))**4 / a**3
norm_a = a
keys = {1: ('h0', 'h2', 'l0', 'l2', 'l1', 'lp'), 2: ('h0', 'h2', 'l0', 'l2', 'l1')}
out(f'Convention sensitivity of the TN36 Love numbers, PREM (fluid core, Coriolis), surface flattening '
    f'1/{1 / f:.1f}; changes vs baseline in units of 1e-5')
for tide, (m, fr) in TIDES.items():
    S = solve(m, fr)
    f0 = raw_fields(S)
    b0 = body(S, *f0)
    rows = {}
    rows['baseline'] = iers_fit_fields(m, th, *(v / norm_e for v in b0))
    if m == 1:
        Ss = solve(m, fr, frame='surface')
        rows['frame: surface'] = iers_fit_fields(m, th, *(v / norm_e for v in body(Ss, *raw_fields(Ss))))
    rows['geodetic latitude'] = iers_fit_fields(m, th, *(v / norm_e for v in b0), f_geodetic=f)
    d = f * np.sin(2 * th)                          # sin 2 phi = sin 2 theta
    ur, ut, up = b0
    nrm = (ur - d * ut, ut - d * ur, up)
    rows['normal direction'] = iers_fit_fields(m, th, *(v / norm_e for v in nrm))
    rows['normal + geodetic'] = iers_fit_fields(m, th, *(v / norm_e for v in nrm), f_geodetic=f)
    f1 = raw_fields(S, deriv=True)
    rig = [x - y for x, y in zip(f0, b0)]           # removed rigid rotation (transverse only)
    db = [x - y / a for x, y in zip(f1, rig)]       # d/dr of the body tide
    delta = -2.0 / 3.0 * f * a * P2c
    Rs = 1 - 2.0 / 3.0 * (f + a * df) * P2c
    ms = [x - delta / Rs * y for x, y in zip(b0, db)]
    rows['mean sphere'] = iers_fit_fields(m, th, *(v / norm_e for v in ms))
    rows['norm. at mean radius'] = iers_fit_fields(m, th, *(v / norm_a for v in b0))
    if m == 2:
        Sn = solve(m, fr, centrifugal=False)
        rows['no centrifugal ref.'] = iers_fit_fields(m, th, *(v / norm_e for v in body(Sn, *raw_fields(Sn))))
    k = keys[m]
    out('')
    out(f'{tide}: ' + ''.join(f'{x:>10s}' for x in k))
    base = rows['baseline']
    out(f"{'baseline (this paper)':26s}" + ''.join(f'{base[x].real:+10.5f}' for x in k))
    for label, r in rows.items():
        if label == 'baseline':
            continue
        out(f'{label:26s}' + ''.join(f'{1e5 * (r[x].real - base[x].real):+10.1f}' for x in k))
open('results/convention_table.txt', 'w').write('\n'.join(lines) + '\n')
