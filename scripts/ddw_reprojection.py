# -*- coding: utf-8 -*-
#
# Paper 1, Online Resource 1 (comparison with Dehant et al.): their Table 10 (eq. 20) re-projected onto
# the definitions of Mathews et al. (their eq. 21) and compared with their Table 11.
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
# 16/10/2026 Machiel Bos
#===============================================================================
"""Is the diurnal l(1) of Dehant et al. (1999) in their solution or in their conversion?

Their eq. (20) describes the surface displacement with h0, h+, l0, l+, w-, w+ (Table 10), their eq. (21)
with the parameters of Mathews et al. (Table 11). Both are functions of colatitude only, so the
conversion is a least-squares projection (fully normalized harmonics). Part 1 rebuilds Table 11 from
Table 10; part 2 projects one term at a time, which shows that w- (rigid rotation) maps only onto l(-)
and that l(1) comes from w+ and l+; part 3 projects our own surface field (elastic hydrostatic PREM,
set-up of scripts/ddw_fingerprint.py, mantle frame) onto eq. (20), so that it can be compared with
Table 10 term by term.
Usage: python scripts/ddw_reprojection.py  ->  results/ddw_reprojection.txt
"""

import numpy as np
from scipy.special import lpmv, factorial
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


th = np.linspace(0.02, np.pi - 0.02, 2000)
c = np.cos(th); s = np.sin(th)
def Y(l, m):
    if l < m: return 0 * th
    N = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m))
    return N * lpmv(m, l, c)
def dY(l, m, d=1e-6):
    if l < m: return 0 * th
    N = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m))
    return N * (lpmv(m, l, np.cos(th + d)) - lpmv(m, l, np.cos(th - d))) / (2*d)
P2 = 0.5*(3*c**2 - 1); P1 = c
def eq20(m, h0, hp, l0, lp, wm, wp, hm=0.0):
    ur = h0*Y(2,m) + hp*Y(4,m) + hm*Y(0,m)
    ut = l0*dY(2,m) + wp*m/s*Y(3,m) + wm*m/s*Y(1,m) + lp*dY(4,m)
    ul = l0*m/s*Y(2,m) + wp*dY(3,m) + wm*dY(1,m) + lp*m/s*Y(4,m)
    return ur, ut, ul
def fit21(m, ur, ut, ul):
    A = np.column_stack([Y(2,m), P2*Y(2,m)] + ([Y(0,m)] if m == 0 else []))
    pr = np.linalg.lstsq(A, ur, rcond=None)[0]
    cols_t = [dY(2,m), P2*dY(2,m), P1*m/s*Y(2,m)] + ([m/s*Y(1,m)] if m == 1 else [])
    cols_l = [m/s*Y(2,m), P2*m/s*Y(2,m), P1*dY(2,m)] + ([dY(1,m)] if m == 1 else [])
    B = np.vstack([np.column_stack(cols_t), np.column_stack(cols_l)])
    pt, res, *_ = np.linalg.lstsq(B, np.concatenate([ut, ul]), rcond=None)
    mis = np.abs(B @ pt - np.concatenate([ut, ul])).max()
    return pr, pt, mis
# Table 10 (elastic hydrostatic): h0, h+, l0, l+, w-, w+
T10 = {'M2': (2, 0.60189, -0.00012, 0.08391, 0.00001, 0.0, 0.00027),
       'O1': (1, 0.59546, -0.00017, 0.08310, 0.00002, 0.00077, 0.00024),
       'P1': (1, 0.57403, -0.00018, 0.08379, 0.00002, 0.00074, 0.00024),
       'K1': (1, 0.51391, -0.00023, 0.08568, 0.00002, 0.00064, 0.00021),
       'psi1': (1, 0.93398, 0.00008, 0.07248, -0.00001, 0.00129, 0.00041),
       'OO1': (1, 0.60085, -0.00016, 0.08297, 0.00001, 0.00078, 0.00026)}
# Table 11: h(0), h(2), l(0), l(1), l(2), l'
T11 = {'M2': (0.60175, -0.00047, 0.08359, 0.00100, 0.00008, None),
       'O1': (0.59553, -0.00048, 0.08298, 0.00070, 0.00008, 0.00020),
       'P1': (0.57411, -0.00053, 0.08367, 0.00070, 0.00009, 0.00018),
       'K1': (0.51400, -0.00065, 0.08558, 0.00060, 0.00011, 0.00015),
       'psi1': (0.93394, 0.00024, 0.07227, 0.00130, -0.00004, 0.00034),
       'OO1': (0.60092, -0.00047, 0.08283, 0.00077, 0.00008, 0.00018)}
OURS_L1 = {'M2': 0.00099, 'O1': 0.00048, 'P1': 0.00045, 'K1': 0.00034, 'psi1': 0.00112, 'OO1': 0.00051}
out(f"{'':5s} {'h(0)':>8s} {'h(2)':>8s} {'l(0)':>8s} {'l(1)':>8s} {'l(2)':>8s} {'l(-)':>8s}   misfit")
for k, (m, *p) in T10.items():
    pr, pt, mis = fit21(m, *eq20(m, *p))
    v = [pr[0], pr[1], pt[0], pt[2], pt[1]] + ([pt[3]] if m == 1 else [np.nan])
    out(f"{k:5s} " + ' '.join(f'{x:+8.5f}' for x in v) + f"   {mis:.1e}   projected from Table 10")
    t = T11[k]
    out(f"{'':5s} " + ' '.join(f'{x:+8.5f}' if x is not None else '        ' for x in t) + f"              Table 11;  ours l(1) {OURS_L1[k]:.5f}")

out()
out('One term at a time (unit value):')
for m in (1, 2):
    for name, p in (('w- = 1', (0, 0, 0, 0, 1.0, 0)), ('w+ = 1', (0, 0, 0, 0, 0, 1.0)), ('l+ = 1', (0, 0, 0, 1.0, 0, 0))):
        if m == 2 and name.startswith('w-'):
            continue
        pr, pt, mis = fit21(m, *eq20(m, *p))
        out(f'  m={m} {name}: l(0) {pt[0]:+.4f} l(2) {pt[1]:+.4f} l(1) {pt[2]:+.4f}' + (f' l(-) {pt[3]:+.4f}' if m == 1 else ''))
out()
out('Our solution in the variables of their eq. (20) (Table 10 below it):')
import warnings
warnings.filterwarnings('ignore')
import sys as _sys
_sys.path.insert(0, 'src')
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.basis import ybar
for tide, m, fr in (('O1', 1, 0.92700), ('M2', 2, 1.4053e-4 / OMEGA_EARTH)):
    md = EarthModel('data/PREM.dat')
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=fr * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    S.solve()
    e = g.eps(md.n_layers - 1, np.array([md.a]))[0]
    norm = (md.a * (1 + e[0] / 3.0))**4 / md.a**3
    C = np.median(ybar(2, m, th, 0)[0] / Y(2, m))
    ur, ut, up = (v / norm / C for v in S.surface_field(th))
    ul = -up                                  # solver: u_lambda x sin(m lam); eq. (20): coefficient of i e_lambda
    pr = np.linalg.lstsq(np.column_stack([Y(2, m), Y(4, m)]), ur, rcond=None)[0]
    ct = [dY(2, m), dY(4, m), m / s * Y(3, m)] + ([m / s * Y(1, m)] if m == 1 else [])
    cl = [m / s * Y(2, m), m / s * Y(4, m), dY(3, m)] + ([dY(1, m)] if m == 1 else [])
    pt = np.linalg.lstsq(np.vstack([np.column_stack(ct), np.column_stack(cl)]), np.concatenate([ut, ul]),
                         rcond=None)[0]
    t = T10[tide]
    out(f'  {tide} ours  h0 {pr[0]:.5f}  h+ {pr[1]:+.5f}  l0 {pt[0]:.5f}  l+ {pt[1]:+.5f}  w+ {pt[2]:+.5f}'
        + (f'  w- {pt[3]:+.5f}' if m == 1 else ''))
    out(f'  {tide} DDW   h0 {t[1]:.5f}  h+ {t[2]:+.5f}  l0 {t[3]:.5f}  l+ {t[4]:+.5f}  w+ {t[6]:+.5f}'
        + (f'  w- {t[5]:+.5f}' if m == 1 else ''))
out()
out('Dictionary in the parameters of the Online Resource (Sect. S8, Schmidt Y_m, unit amplitude A):')
# Fit DDW eq. (20), one term at a time, to R_m, N_m, E_m of Sect. S8 written in latitude phi.
# V_2m/g = k A with k = Y_m(phi)/Y_2m(theta); u_N = -u_theta, u_E = u_lambda without its factor i.
sp, cp = c, s                                   # sin(phi) = cos(theta), cos(phi) = sin(theta)
P2s = 0.5*(3*sp**2 - 1)
Ym = {0: P2s, 1: np.sqrt(3)*sp*cp, 2: np.sqrt(3)/2*cp**2}
def ours(m):
    """Columns of S9-S11: radial (h0, h2, h') and horizontal (l0, l2, l1, l')."""
    Y_ = Ym[m]; r = [Y_, P2s*Y_] + ([np.sqrt(4*np.pi/5) + 0*sp] if m == 0 else [])
    if m == 0:
        N = [3*sp*cp, P2s*3*sp*cp, 0*sp, 0*sp]
        E = [0*sp, 0*sp, -3*sp**2*cp, np.sqrt(4*np.pi/5)*cp]
    elif m == 1:
        N = [np.sqrt(3)*np.cos(2*np.arcsin(sp)), P2s*np.sqrt(3)*np.cos(2*np.arcsin(sp)), -np.sqrt(3)*sp**2,
             np.sqrt(8*np.pi/5) + 0*sp]
        E = [np.sqrt(3)*sp, P2s*np.sqrt(3)*sp, -np.sqrt(3)*sp*np.cos(2*np.arcsin(sp)), -np.sqrt(8*np.pi/5)*sp]
    else:
        N = [-np.sqrt(3)*sp*cp, -P2s*np.sqrt(3)*sp*cp, -np.sqrt(3)*sp*cp]
        E = [np.sqrt(3)*cp, P2s*np.sqrt(3)*cp, np.sqrt(3)*sp**2*cp]
    return np.column_stack(r), np.vstack([np.column_stack(N), np.column_stack(E)])
names = ['h0', 'h+', 'h-', 'l0', 'l+', 'w+', 'w-']
for m in (0, 1, 2):
    k = np.median(Ym[m] / Y(2, m))
    Ar, Ah = ours(m)
    hn = ['h(0)', 'h(2)'] + (["h'"] if m == 0 else [])
    ln = ['l(0)', 'l(2)', 'l(1)'] + (["l'"] if m < 2 else [])
    for j, nm in enumerate(names):
        if (nm == 'h-' and m != 0) or (nm == 'w-' and m == 2):
            continue
        p = [0.0]*7; p[j] = 1.0
        h0, hp, hm, l0, lp, wp, wm = p
        ur = h0*Y(2,m) + hp*Y(4,m) + hm*Y(0,m)
        ut = l0*dY(2,m) + lp*dY(4,m) + m/s*(wp*Y(3,m) + wm*Y(1,m))
        ul = m/s*(l0*Y(2,m) + lp*Y(4,m)) + wp*dY(3,m) + wm*dY(1,m)
        xr, *_ = np.linalg.lstsq(Ar, k*ur, rcond=None)
        rhs = np.concatenate([-k*ut, k*ul])
        xh, *_ = np.linalg.lstsq(Ah, rhs, rcond=None)
        mis = max(np.abs(Ar @ xr - k*ur).max(), np.abs(Ah @ xh - rhs).max())
        terms = [f'{a} {v:+.4f}' for a, v in list(zip(hn, xr)) + list(zip(ln, xh)) if abs(v) > 5e-5]
        out(f'  m={m} {nm} = 1:  ' + '  '.join(terms) + f'   (misfit {mis:.0e})')
open('results/ddw_reprojection.txt', 'w').write('\n'.join(lines) + '\n')
