# -*- coding: utf-8 -*-
#
# Paper 1: size of the rigid rotation removed from the diurnal displacement (the nutation seen from
# the rotating Earth) and of the difference between the mantle and the surface frame, in mm.
#
# This file is part of earth-tides 1.1.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 15/10/2026 Machiel Bos
#===============================================================================
"""Paper 1, Sect. 3.5 and Online Resource 1 (conventions): how large the frame question is.

PREM, m = 1, fluid core, Coriolis, inertia, consistent reference state (as scripts/convention_table.py).
For each diurnal tide l' is fitted (TN36 form) to the surface displacement (a) without removing any
rigid rotation, (b) in the mantle (Tisserand) frame, (c) in Wahr's surface frame. (a) - (b) is the
rotation removed; it is the forced nutation seen from the rotating Earth (O1: 13.66-day nutation,
~0.09" -> ~2.8 m; P1: semiannual, ~0.52" -> ~16 m; unbounded at K1, the tilt-over mode).
Millimetres: northward displacement at the equator, where the l' term (north = l' sqrt(24 pi/5) C)
is largest, for the Tamura amplitude of the tide; l0 north shown for comparison.
Usage: python scripts/frame_size.py  ->  results/frame_size.txt
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, ybar, cheb_basis
from earthtides.iers import iers_fit_fields
from earthtides.etide import Tamura, W_functions

md = EarthModel('data/PREM.dat')
geo = ClairautGeometry(md)
th = np.linspace(0.03, np.pi - 0.03, 600)
a, iS = md.a, md.n_layers - 1
f = geo.eps(iS, np.array([a]))[0][0]
norm_e = (a * (1 + f / 3.0))**4 / a**3
# name: (cycles per sidereal day, cycles per solar day)
TIDES = {'O1': (0.92700, 0.929536), 'P1': (0.99452, 0.997262), 'psi1': (1.00548, 1.005476)}


def solve(fr, frame):
    S = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=fr * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame=frame)
    S.solve()
    return S


def raw_fields(S):
    """u_r, u_theta, u_lambda at s = a, no rigid-rotation removal."""
    N, nb = S.N, S.nfld * S.N
    Tp = cheb_basis(np.array([1.0]), N)[0]
    v = {fld: float(S.coef[iS * nb + j * N: iS * nb + (j + 1) * N] @ Tp[:, 0])
         for j, fld in enumerate(S.fields)}
    ur = sum(v[('U', n)] * ybar(n, 1, th, 0)[0] for n in S.deg)
    ut, up = 0 * th, 0 * th
    for n in S.degV:
        ut, up = ut + v[('V', n)] * ybar(n, 1, th, 1)[1], up - v[('V', n)] * ybar(n, 1, th, 0, over_sin=True)[0]
    for n in S.degW:
        ut, up = ut - v[('W', n)] * ybar(n, 1, th, 0, over_sin=True)[0], up + v[('W', n)] * ybar(n, 1, th, 1)[1]
    return ur, ut, up


def fit(S, remove):
    ur, ut, up = raw_fields(S)
    if remove:
        ut, up = (S._remove_mantle_rotation(th, ut, up) if S.frame == 'mantle'
                  else S._remove_rigid_rotation(th, ut, up))
    return iers_fit_fields(1, th, ur / norm_e, ut / norm_e, up / norm_e)


def main():
    cat = Tamura()
    s = np.flatnonzero((cat.n == 2) & (cat.m == 1))
    dW = abs(W_functions(2, 1, 0.0)[1])          # north per unit amplitude at the equator, l = 1
    c1 = np.sqrt(24 * np.pi / 5) / 3              # l' north relative to l0 north at the equator
    lines = ['Rigid rotation of the diurnal displacement, PREM; l\' (TN36) and northward displacement '
             'at the equator [mm]',
             f"{'tide':5s}{'l0':>9s}{'l\' none':>12s}{'l\' mantle':>12s}{'l\' surface':>12s}"
             f"{'removed mm':>13s}{'mant-surf mm':>14s}{'l0 north mm':>13s}"]
    for name, (fr, fcpd) in TIDES.items():
        A = abs(cat.A[s[np.argmin(abs(cat.f_cpd[s] - fcpd))]])
        Sm, Ss = solve(fr, 'mantle'), solve(fr, 'surface')
        r0, rm, rs = fit(Sm, False), fit(Sm, True), fit(Ss, True)
        lp0, lpm, lps, l0 = (x.real for x in (r0['lp'], rm['lp'], rs['lp'], rm['l0']))
        mm = lambda x: abs(x) * c1 * dW * A * 1e3
        lines.append(f'{name:5s}{l0:9.5f}{lp0:+12.5f}{lpm:+12.5f}{lps:+12.5f}'
                     f'{mm(lp0 - lpm):13.1f}{mm(lpm - lps):14.3f}{abs(l0) * dW * A * 1e3:13.2f}')
    print('\n'.join(lines))
    open('results/frame_size.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
