# -*- coding: utf-8 -*-
#
# Benchmark: Greff-Lefftz, Metivier & Legros (2005, Celest. Mech. Dyn. Astr.
# 93, 113-146), analytical Love numbers of a homogeneous, incompressible,
# self-gravitating, hydrostatically flattened elastic Earth, semidiurnal (m =
# 2), static.
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
# 6/10/2026 Machiel Bos
#===============================================================================
"""Benchmark: Greff-Lefftz, Metivier & Legros (2005, Celest. Mech. Dyn. Astr. 93, 113-146),
analytical Love numbers of a homogeneous, incompressible, self-gravitating, hydrostatically
flattened elastic Earth, semidiurnal (m = 2), static.

Their constants imply mu_bar = 19 mu / (2 rho g a) = 3.15592 (from h2 = 5/2/(1+mu_bar));
we scale mu to the same mu_bar so that the spherical Love numbers coincide, and approach
incompressibility with lambda/mu = 1e3, 1e4, 1e5 (linear extrapolation in mu/lambda).
Compared: normal and tangential displacement on the ellipsoid in their ellipsoidal basis
(their eqs 48-52) and the exterior potential coefficients (k2 + dk2, dk4; eqs 37-39).
Run:  python greff2005.py   ->  greff2005.out  (model file HOMOGI.dat is written here)
"""
import math
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
from earthtides.model import EarthModel, G
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, mapping
from earthtides.solver import TidalSolver, ybar

HERE = Path(__file__).resolve().parent
RHO, A = 5520.0, 6371e3
MUBAR = 2.5 / 0.601553 - 1.0
g0 = 4 / 3 * math.pi * G * RHO * A
MU = MUBAR * 2 * RHO * g0 * A / 19


def make(N):
    vs = math.sqrt(MU / RHO) / 1e3
    vp = math.sqrt((N + 2) * MU / RHO) / 1e3
    blk = lambda name, v: f"\n{name}\n{'-' * len(name)}\n6371.0  {v:16.10f}    0.00000    0.00000    0.00000\n"
    s = "HOMOGI\t| homogeneous incompressible test\n1       | Number of layers\n" + blk('rho', RHO / 1e3) \
        + blk('Vp_h', vp) + blk('Vp_v', vp) + blk('Vs_h', vs) + blk('Vs_v', vs) + blk('eta', 1.0) \
        + "\nQmu\n---\n6371.0   600.0\n"
    (HERE / 'HOMOGI.dat').write_text(s)
    return EarthModel(str(HERE / 'HOMOGI.dat'))


th = np.linspace(0.03, np.pi - 0.03, 401)
c4 = math.sqrt(3) / 6 / math.sqrt(4 / 720)       # V0 P42 -> Schmidt Ybar42 factor
c3 = math.sqrt(3) / 6 / math.sqrt(2 / 120)       # V0 P32 -> Ybar32
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def compare(md, orot):
    """Ellipsoid quantities of their eqs (48)-(53) from our surface fields."""
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=2, K=6, N=16, omega_f=0.0, omega_rot=orot)
    r = S.solve()
    a, gs = md.a, md.g_surface
    ur, ut, up = (v / a for v in S.surface_field(th))
    mp = mapping(a, th, *g.eps(0, np.array([a])))
    R, Rt = mp['R'], mp['R_t']
    hh = np.hypot(R, Rt)
    nr, nt = R / hh, -Rt / hh
    un, uS = ur * nr + ut * nt, -ur * nt + ut * nr
    Y = {n: ybar(n, 2, th, 1) for n in (2, 3, 4)}
    Ys = {n: 2 * ybar(n, 2, th, 0, over_sin=True)[0] for n in (2, 3, 4)}
    hn = np.linalg.lstsq(np.column_stack([Y[2][0], Y[4][0]]), un, rcond=None)[0]
    At = np.vstack([np.column_stack([Y[2][1], Y[4][1], -Ys[3]]),
                    np.column_stack([-Ys[2], -Ys[4], Y[3][1]])])
    lt = np.linalg.lstsq(At, np.concatenate([uS, up]), rcond=None)[0]
    # geoid on the deformed outer surface: (phi1 + phiT)(R(a,theta)) / (-g a)   (their eq. 53)
    ph1 = sum(-r['k'][n] * gs * a * (a / R)**(n + 1) * ybar(n, 2, th, 0)[0] for n in r['k'])
    geo = (ph1 - gs * a * (R / a)**2 * Y[2][0]) / (-gs * a)
    kk = np.linalg.lstsq(np.column_stack([Y[2][0], Y[4][0]]), geo, rcond=None)[0]
    return dict(hn=hn[0], hp=hn[1] / c4, ln=lt[0], lp=lt[1] / c4, ls=lt[2] / c3,
                kn=kk[0] - 1.0, kp=kk[1] / c4)


if __name__ == '__main__':
    md = make(100000)                              # lambda = 1e5 mu: incompressible to ~1e-6
    r0 = TidalSolver(md, m=2, K=2, N=16, omega_f=0.0).solve()
    GR = dict(hn=0.603140, hp=-0.000332, ln=0.181544, lp=-0.000064, ls=-0.000224,
              kn=0.363824, kp=-0.000633)
    names = dict(hn='h2+ho (normal)', hp='h+', ln='l2+lo (tangential)', lp='l+', ls='l* (toroidal)',
                 kn='k2+ko (geoid)', kp='k+ (geoid)')
    out('Greff-Lefftz et al. (2005): homogeneous, incompressible (lambda = 1e5 mu), mu_bar = %.5f, '
        '1/alpha = %.1f, static, m = 2' % (MUBAR, 1 / ClairautGeometry(md).flattening_surface))
    out(f"  sphere: h2 = {r0['h']:.6f} (0.601553)  l2 = {r0['l']:.6f} (0.180466)  k2 = {r0['kk']:.6f} (0.360932)")
    A_ = compare(md, 0.0)
    B_ = compare(md, OMEGA_EARTH)
    out(f"  {'':22s}{'no centrifugal':>16s}{'with centrifugal':>18s}{'Greff et al.':>14s}")
    for k in GR:
        out(f'  {names[k]:22s}{A_[k]:+16.6f}{B_[k]:+18.6f}{GR[k]:+14.6f}')
    out('  "no centrifugal": reference gravity in the perturbation equations purely gravitational,')
    out('  which reproduces their formulation; "with": centrifugal force on the displaced parcel included.')
    (HERE / 'greff2005.out').write_text('\n'.join(lines) + '\n')
