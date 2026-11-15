# -*- coding: utf-8 -*-
#
# Free core nutation of Hough's (1895) Earth with a viscous core: Ekman damping
# (Greenspan 1968) as a known answer for the no-slip core.
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""Free core nutation of Hough's Earth (ex5) with a viscous, no-slip core.

The rigid shell and homogeneous fluid core of ex5, now with a Newtonian viscosity in the core
(EarthModel.viscous_core: mu = i omega eta, Ekman sub-layers below the CMB). A no-slip wall adds
a viscous torque on the core, which enters the FCN as e_f -> e_f + K (Sasao et al. 1980; Mathews
et al. 2002):
    sigma_FCN = 1 + (A / A_m) (e_f + K),   K = (c_r + i c_i) sqrt(E),   E = nu / (Omega r_c^2),
and sqrt(E) = delta / r_c with delta = sqrt(nu / Omega) the Ekman thickness. c_r + i c_i follows
from the Ekman-layer stress integrated over a spherical wall (ekman_coefficient below; it shares no
code with the solver); for sigma = 1 it is 0.2585 + 2.6205 i, the spin-over correction of
Greenspan (1968). Im K > 0 is the damping (time dependence exp(i omega t)).

The free mode is the complex frequency where the solver's matrix H(sigma) is singular: Muller's
method on det H, with the rows scaled once at the starting frequency so that det H stays analytic.
The inviscid root (ex5) is the reference; the solver's shift (root - inviscid) / ((A/A_m) sqrt(E))
is compared with c_r + i c_i at the root's frequency.
Run:  python ekman.py   ->  ekman.out   (about 10 minutes on 5 cores)
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
from scipy.integrate import quad

from earthtides.geometry import ExactSpheroidGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ex5'))
import hough1895 as ch                                   # noqa: E402

RC, A = ch.RC * 1e3, ch.A * 1e3
A_RATIO = 1.0 / (1.0 - (RC / A)**5)                      # A / A_m for similar homogeneous layers
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def ekman_coefficient(sig=1.0):
    """K / sqrt(E) of the spin-over flow u = w x r, w = (1, i, 0), in a sphere with a no-slip wall,
    frequency sig Omega in the frame of the wall. Boundary layer: nu q'' = i (omega +- f) q for
    q = u_theta +- i u_phi, f = 2 Omega cos(theta); wall traction -rho sqrt(nu Omega) sqrt(i (sig +- 2
    cos theta)) Q; torque I gamma w with I = 8 pi rho r^5 / 15; K = -i gamma / Omega."""
    ph = np.linspace(0.0, 2 * np.pi, 65)[:-1]
    w = np.array([1.0, 1j, 0.0])[:, None]

    def f(th, part):
        st, ct = np.sin(th), np.cos(th)
        r = np.array([st * np.cos(ph), st * np.sin(ph), ct + 0 * ph])
        et = np.array([ct * np.cos(ph), ct * np.sin(ph), -st + 0 * ph])
        ep = np.array([-np.sin(ph), np.cos(ph), 0 * ph])
        u = np.cross(w.T, r.T).T
        ut, up = (u * et).sum(0), (u * ep).sum(0)
        tp = -np.sqrt(1j * (sig + 2 * ct) + 0j) * (ut + 1j * up)
        tm = -np.sqrt(1j * (sig - 2 * ct) + 0j) * (ut - 1j * up)
        t = (tp + tm) / 2 * et + (tp - tm) / 2j * ep
        v = (np.cross(r.T, t.T).T * np.conj(w)).sum(0).mean() * 2 * np.pi * st
        return v.real if part == 0 else v.imag
    pts = [np.arccos(sig / 2), np.arccos(-sig / 2)] if sig < 2 else None
    g = sum(c * quad(f, 0.0, np.pi, args=(k,), points=pts, limit=400)[0] for k, c in ((0, 1), (1, 1j)))
    return -1j * g / (8 * np.pi / 15 * 2)


def fcn_root(md0, fl, delta, K, N, start, Om=OMEGA_EARTH, tol=1e-12, itmax=40):
    """Complex FCN (cycles per sidereal day) of md0 with a viscous core of Ekman thickness delta
    (m; 0: inviscid), from Muller's method on det H started at the three points `start`."""
    g = ExactSpheroidGeometry(fl)

    def H(s):
        md = md0.viscous_core(s * Om, delta=delta) if delta > 0 else md0
        S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=Om, omega_t=s * Om, omega_rot=Om,
                        wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle')
        S.solve()
        return S.H.astype(complex)
    H0 = H(start[1])
    sc = 1.0 / np.abs(H0).max(axis=1)
    ref = np.linalg.slogdet(H0 * sc[:, None])[1]

    def d(s):
        sg, ld = np.linalg.slogdet(H(s) * sc[:, None])
        return sg * np.exp(ld - ref)
    xs = list(start)
    fs = [d(x) for x in xs]
    for _ in range(itmax):
        (x0, x1, x2), (f0, f1, f2) = xs[-3:], fs[-3:]
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        D = np.sqrt(b * b - 4 * a * f2 + 0j)
        x3 = x2 - 2 * f2 / (b + D if abs(b + D) > abs(b - D) else b - D)
        xs.append(x3)
        fs.append(d(x3))
        if abs(x3 - x2) < tol:
            break
    return xs[-1]


def job(args):
    delta, K, N = args
    md = ch.make(path=HERE / 'HOUGH.dat')
    fl = ch.maclaurin_flattening(OMEGA_EARTH)
    p = ch.prediction(fl)
    d = p - 1
    if delta == 0:
        return args, fcn_root(md, fl, 0.0, K, N, [p - 0.02 * d, p + 0.02 * d, p + 0.01 * d])
    e = A_RATIO * delta / RC * ekman_coefficient(1.0)
    return args, fcn_root(md, fl, delta, K, N, [p + 0.97 * e - 0.02 * d, p + 1.03 * e + 0.02 * d,
                                                 p + e + 0.01j * d])


RUNS = [(0.0, 12, 24), (4e3, 12, 24), (2e3, 12, 24), (1e3, 12, 24), (5e2, 12, 32), (2.5e2, 12, 40),
        (1e3, 6, 24), (1e3, 8, 24), (1e3, 10, 24), (1e3, 14, 24), (1e3, 16, 24)]

if __name__ == '__main__':
    ch.make(path=HERE / 'HOUGH.dat')
    with Pool(6) as pool:
        res = dict(pool.map(job, RUNS, chunksize=1))
    s0 = res[RUNS[0]].real
    out("Hough's Earth with a viscous core: FCN (cycles per sidereal day, mantle frame) and the Ekman")
    out(f"correction K / sqrt(E) = (root - inviscid) / ((A/A_m) delta / r_c), A/A_m = {A_RATIO:.5f}")
    out(f"inviscid root (K 12, N 24) {s0:.9f}; Hough's formula {ch.prediction(ch.maclaurin_flattening(OMEGA_EARTH)):.9f}")
    out(f"{'delta km':>8s} {'K':>3s} {'N':>3s} {'root':>22s} {'solver K/sqrt(E)':>17s} {'Ekman (Greenspan)':>18s} {'|diff|/|K|':>10s}")
    for r in RUNS[1:]:
        s = res[r]
        k = (s - s0) * RC / (A_RATIO * r[0])
        ke = ekman_coefficient(s.real)
        out(f"{r[0] / 1e3:8.2f} {r[1]:3d} {r[2]:3d} {s.real:.9f}{s.imag:+.3e}i {k.real:7.4f}{k.imag:+8.4f}i "
            f"{ke.real:8.4f}{ke.imag:+8.4f}i {abs(k - ke) / abs(ke):10.1e}")
    out(f"Greenspan (1968), sigma = 1: {ekman_coefficient(1.0):.4f}")
    (HERE / 'ekman.out').write_text('\n'.join(lines) + '\n')
