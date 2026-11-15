# -*- coding: utf-8 -*-
#
# The degree-3 toroidal term w+ of Dehant et al. (1999) against ours: line by
# line, by cause, by order.
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
"""The degree-3 toroidal term w+ of Dehant et al. (1999) against ours: line by line, by cause, by order.

Our elastic hydrostatic PREM (set-up of scripts/ddw_fingerprint.py, mantle frame) projected onto their
eq. (20). Part 1: w+ at the lines of their Table 10, split as in their Table 4 into the geometric
term (the horizontal degree-2 field evaluated on the flattened surface instead of the mean sphere,
-(2/3) eps P2 r dV/dr with r dV/dr = V - U at a traction-free surface, i.e. l0 - h0) and the toroidal
field W3 at the mean radius (the rest). Part 2: w+ with the flattening (geometry scale)
and the Coriolis force (rate factor) scaled separately by 0, 0.5, 1, 2, which splits it into the
linear parts, the second-order parts and the cross term; a first-order theory keeps only the former.
Part 3: their resonance formula for w+ (eq. 23, Tables 13 and 14b) against ours, and the horizontal
displacement of the offset (H times delta w+ times the degree-3 toroidal pattern; |V21/g| = H).
Usage: python scripts/ddw_wplus.py  ->  results/ddw_wplus.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import lpmv, factorial

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.basis import ybar

th = np.linspace(0.02, np.pi - 0.02, 2000)
c, s = np.cos(th), np.sin(th)
lines = []
md = EarthModel('data/PREM.dat')
# Dehant et al. (1999), Table 10 (elastic hydrostatic PREM): w+, h0, l0; frequencies in cycles per sidereal day
DDW = {'O1': (0.92700, 0.00024, 0.59546, 0.08310), 'P1': (0.99454, 0.00024, 0.57403, 0.08379),
       'K1': (1.0 + 1e-6, 0.00021, 0.51391, 0.08568), 'psi1': (1.00273, 0.00041, 0.93398, 0.07248),
       'OO1': (1.07300, 0.00026, 0.60085, 0.08297), 'M2': (1.4053e-4 / OMEGA_EARTH, 0.00027, 0.60189, 0.08391)}


def out(t=''):
    print(t, flush=True)
    lines.append(t)


def Y(l, m):
    return np.sqrt((2*l + 1)/(4*np.pi)*factorial(l - m)/factorial(l + m))*lpmv(m, l, c)


def dY(l, m, d=1e-6):
    N = np.sqrt((2*l + 1)/(4*np.pi)*factorial(l - m)/factorial(l + m))
    return N*(lpmv(m, l, np.cos(th + d)) - lpmv(m, l, np.cos(th - d)))/(2*d)


def fit20(m, ut, ul):
    ct = [dY(2, m), dY(4, m), m/s*Y(3, m)] + ([m/s*Y(1, m)] if m == 1 else [])
    cl = [m/s*Y(2, m), m/s*Y(4, m), dY(3, m)] + ([dY(1, m)] if m == 1 else [])
    return np.linalg.lstsq(np.vstack([np.column_stack(ct), np.column_stack(cl)]), np.concatenate([ut, ul]),
                           rcond=None)[0]


def geometric(m, eps, h0, l0):
    """w+ of the degree-2 horizontal field moved from the mean sphere to the surface r = a(1 - 2/3 eps P2)."""
    d = -(2/3)*eps*0.5*(3*c**2 - 1)*(l0 - h0)
    return fit20(m, d*dY(2, m), d*m/s*Y(2, m))[2]


def wplus(m, fr, scale=1.0, cfac=1.0, full=False):
    """w+ of eq. (20) for our solution; scale multiplies the flattening, cfac the Coriolis rate."""
    g = ClairautGeometry(md, scale=scale)
    S = TidalSolver(md, g, m=m, K=4, N=24, omega_f=0.0, coriolis=cfac*OMEGA_EARTH, omega_t=fr*OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    S.solve()
    e = g.eps(md.n_layers - 1, np.array([md.a]))[0]
    norm = (md.a*(1 + e[0]/3.0))**4/md.a**3
    C = np.median(ybar(2, m, th, 0)[0]/Y(2, m))
    ur, ut, up = (v/norm/C for v in S.surface_field(th))
    ul = -up                                  # solver: u_lambda x sin(m lam); eq. (20): coefficient of i e_lambda
    p = fit20(m, ut, ul)
    if not full:
        return p[2]
    h0 = np.linalg.lstsq(np.column_stack([Y(2, m), Y(4, m)]), ur, rcond=None)[0][0]
    return p[2], h0, p[0], e[0]


def main():
    out('Part 1: w+ line by line (x 1e5) = geometric term + W3 at the mean radius (their Table 4);')
    out('        "no Coriolis" = our w+ without the Coriolis force')
    for tide, (fr, wd, hd, ld) in DDW.items():
        m = 2 if tide == 'M2' else 1
        (w, h0, l0, eps), w0 = wplus(m, fr, full=True), wplus(m, fr, 1.0, 0.0)
        g, gd = geometric(m, eps, h0, l0), geometric(m, eps, hd, ld)
        out(f'  {tide:5s} ours {w*1e5:6.2f} = {g*1e5:6.2f} + {(w - g)*1e5:5.2f}   DDW {wd*1e5:4.1f} = {gd*1e5:6.2f} + '
            f'{(wd - gd)*1e5:5.2f}   DDW - ours {(wd - w)*1e5:+5.2f}   no Coriolis {w0*1e5:6.2f}')
    out('  (DDW Table 4, M2: geometric 0.00026 and W3 0.00006 in their normalization, x (5/7)^1/2 = 0.00022, 0.00005)')
    out()
    out('Part 2: w+ (x 1e5) with the flattening x (0, 0.5, 1, 2) [rows] and the Coriolis force x (0, 0.5, 1, 2)')
    for tide in ('O1', 'M2'):
        m, fr = (2 if tide == 'M2' else 1), DDW[tide][0]
        W = {(a, b): wplus(m, fr, a, b) for a in (0.0, 0.5, 1.0, 2.0) for b in (0.0, 0.5, 1.0, 2.0)}
        out(f'  {tide}')
        for a in (0.0, 0.5, 1.0, 2.0):
            out(f'    f x {a:3.1f}: ' + '  '.join(f'{W[a, b]*1e5:+8.3f}' for b in (0.0, 0.5, 1.0, 2.0)))
        lin_f = 2*(W[0.5, 0] - W[0, 0]); quad_f = W[1, 0] - W[0, 0] - lin_f
        lin_c = 2*(W[0, 0.5] - W[0, 0]); quad_c = W[0, 1] - W[0, 0] - lin_c
        cross = W[1, 1] - W[1, 0] - W[0, 1] + W[0, 0]
        out(f'    at the Earth: flattening linear {lin_f*1e5:+.2f}, squared {quad_f*1e5:+.2f}; Coriolis linear '
            f'{lin_c*1e5:+.2f}, squared {quad_c*1e5:+.2f}; cross {cross*1e5:+.2f}; neither {W[0, 0]*1e5:+.2f}; '
            f'total {W[1, 1]*1e5:+.2f}; second order {(quad_f + quad_c + cross)*1e5:+.2f}')
    out()
    out('Part 3: their eq. (23) with Table 14b (w+, background b0 + 54-min modes, FCN term) against ours (x 1e5)')
    lam = dict(O1=-0.9269978, CW=0.0024865, FCN=-1.0021832, p54=-27.2010192, m54=26.4068484)   # Table 13
    b0, ar = 0.000239, dict(CW=-3.0126e-9, FCN=-1.1832e-6, p54=0.0014866, m54=-0.0010977)        # Table 14b
    offs = []
    for tide in ('O1', 'P1', 'K1', 'psi1', 'OO1'):
        om = -DDW[tide][0]                                   # their sign of the frequency
        term = {j: ar[j]/(lam[j] - om)*(om - lam['O1']) for j in ar}
        f23, w = b0 + sum(term.values()), wplus(1, DDW[tide][0])
        offs.append(f23 - w)
        out(f'  {tide:5s} eq. 23 {f23*1e5:6.2f}  (FCN term {term["FCN"]*1e5:+6.2f})   ours {w*1e5:6.2f}   offset {(f23 - w)*1e5:5.2f}')
    th3 = np.linspace(1e-4, np.pi - 1e-4, 20001)
    Nn = np.sqrt(7/(4*np.pi)*factorial(2)/factorial(4))
    Y3 = Nn*lpmv(1, 3, np.cos(th3)); pat = np.hypot(Y3/np.sin(th3), np.gradient(Y3, th3))
    H = {'K1': 0.36878, 'O1': 0.26221, 'P1': 0.12203, 'Q1': 0.05020}          # Cartwright-Tayler, m
    dw = float(np.mean(offs)); rms = np.sqrt(sum(h**2 for h in H.values())/2)
    lat = 90 - np.degrees(th3)
    out(f'  mean offset {dw*1e5:.2f}e-5 -> horizontal displacement (pure toroidal, largest at the poles):')
    out(f'    K1 alone max {H["K1"]*dw*pat.max()*1e3:.3f} mm; all lines rms {rms*dw*pat.max()*1e3:.3f} mm max, '
        f'{rms*dw*np.interp(45, lat[::-1], pat[::-1])*1e3:.3f} mm at 45 deg; in phase {sum(H.values())*dw*pat.max()*1e3:.3f} mm max')
    open('results/ddw_wplus.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
