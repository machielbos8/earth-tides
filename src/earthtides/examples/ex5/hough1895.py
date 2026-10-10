# -*- coding: utf-8 -*-
#
# Free core nutation of Hough's (1895) Earth: rigid shell, homogeneous fluid
# core.
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
"""Free core nutation of Hough's (1895) Earth: rigid shell, homogeneous fluid core.

Homogeneous, self-gravitating, rotating body (rho = 5500 kg/m3) with the exact
equilibrium shape for its rotation rate (Maclaurin spheroid); fluid core out to
r_c = 3480 km, mantle 1e4-1e5 times stiffer than PREM. All layers similar
spheroids. Coriolis + inertia, free rotations, Poisson-consistent reference
potential. With equal densities there is no internal gravitational torque, and
the classical rigid-mantle result applies (Hough 1895; Poincare 1910; e.g.
Sasao et al. 1980 eq. with beta = 0): in the mantle frame the free core
nutation has frequency
    sigma = Omega [1 + (A / A_m) e_f],
e_f = (a^2 - c^2)/(a^2 + c^2) the core's dynamical ellipticity and, for similar
homogeneous layers, A_m / A = 1 - (r_c/a)^5. The formula is first order in e_f.

The eigenfrequency is where the solver's system matrix H(omega) is singular:
det H changes sign (Brent's method). A last line repeats the Earth case without
any reference potential (gravity=False: no centrifugal reference state either).
A second sign change near +17 % of the offset is a numerical artefact: it
moves with N (K=6: +17 % at N=16, +21 % at N=24) and with the core's bulk
modulus (+32 % at Vp = 3000 km/s, gone at 10000 km/s); a homogeneous
compressible fluid under gravity is slightly unstably stratified. The FCN root
stays within 3e-4 of the prediction in all these variants.
Run:  python hough1895.py   ->  hough1895.out  (model file HOUGH.dat is written here)
"""
import math
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import brentq

from earthtides.model import EarthModel, G
from earthtides.geometry import ExactSpheroidGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver

HERE = Path(__file__).resolve().parent
RC, A, RHO = 3480.0, 6371.0, 5500.0
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def make(vs_m=1000.0, vp_c=1000.0, path=HERE / 'HOUGH.dat'):
    """Two layers, equal density; Vs, Vp in km/s."""
    def blk(name, core, mant):
        return (f"\n{name}\n{'-' * len(name)}\n{A:.1f}  {core:14.6f}  0.0  0.0  0.0\n"
                f"{A - RC:.1f}  {mant:14.6f}  0.0  0.0  0.0\n")
    s = ("HOUGH\t| rigid mantle, homogeneous fluid core (Hough 1895)\n2       | Number of layers\n"
         + blk('rho', RHO / 1e3, RHO / 1e3) + blk('Vp_h', vp_c, 2 * vs_m) + blk('Vp_v', vp_c, 2 * vs_m)
         + blk('Vs_h', 0.0, vs_m) + blk('Vs_v', 0.0, vs_m) + blk('eta', 1, 1)
         + f"\nQmu\n---\n{A:.1f}   600.0\n{A - RC:.1f}   600.0\n")
    Path(path).write_text(s)
    return EarthModel(str(path))


def maclaurin_flattening(Om):
    q = Om**2 / (math.pi * G * RHO)
    F = lambda e: (2 * math.sqrt(1 - e * e) * (3 - 2 * e * e) * math.asin(e) / e**3
                   - 6 * (1 - e * e) / e**2 - q)
    e = brentq(F, 1e-6, 0.9)
    return 1 - math.sqrt(1 - e * e)


def prediction(fl):
    g = ExactSpheroidGeometry(fl)
    ef = (g.A**2 - g.C**2) / (g.A**2 + g.C**2)
    return 1 + ef / (1 - (RC / A)**5)


def logdet(md, g, fr, Om, K, N, gravity=True):
    S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=Om, omega_t=fr * Om,
                    omega_rot=Om, wahr_frame=True, dyn_fluid=True, gravity=gravity,
                    consistent_reference=gravity, frame='mantle')
    S.solve()
    return np.linalg.slogdet(S.H / np.abs(S.H).max(axis=1)[:, None])


def fcn(md, fl, Om, K=6, N=16, gravity=True, lo=0.6, hi=1.4, n=9):
    """All eigenfrequencies between lo and hi times the predicted offset."""
    g = ExactSpheroidGeometry(fl)
    p = prediction(fl) - 1
    ref = logdet(md, g, 1 + p, Om, K, N, gravity)[1]
    f = lambda x: (lambda sd: sd[0] * np.exp(sd[1] - ref))(logdet(md, g, 1 + x * p, Om, K, N, gravity))
    xs = np.linspace(lo, hi, n)
    v = [f(x) for x in xs]
    return [1 + p * brentq(f, xs[i], xs[i + 1], xtol=1e-10)
            for i in range(n - 1) if np.sign(v[i]) != np.sign(v[i + 1])]


if __name__ == '__main__':
    md = make()
    out("Hough's Earth (Maclaurin spheroid, rigid shell, fluid core): FCN in cycles per sidereal day")
    out(f"{'Omega':>8s} {'1/f':>8s} {'K':>2s} {'predicted':>12s} {'solver':>12s} {'P pred':>8s} {'P solv':>8s} {'rel diff':>9s}")
    for sc, K in [(2.0, 6), (1.0, 6), (1.0, 8), (0.5, 6)]:
        Om = OMEGA_EARTH * sc
        fl = maclaurin_flattening(Om)
        p = prediction(fl)
        for s in fcn(md, fl, Om, K=K):
            rel = (s - p) / (p - 1)
            out(f"{sc:6.1f}xE {1 / fl:8.2f} {K:2d} {p:12.8f} {s:12.8f} {1 / (p - 1):8.2f} {1 / (s - 1):8.2f}"
                f" {rel:+9.1e}" + ('   numerical mode (see README)' if abs(rel) > 0.05 else '   FCN'))
    fl = maclaurin_flattening(OMEGA_EARTH)
    s = fcn(md, fl, OMEGA_EARTH, gravity=False)
    out(f"without reference potential (no centrifugal pre-stress), Omega_E: "
        + ', '.join(f"{(x - prediction(fl)) / (prediction(fl) - 1):+.1e}" for x in s))
    (HERE / 'hough1895.out').write_text('\n'.join(lines) + '\n')
