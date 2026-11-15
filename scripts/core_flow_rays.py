# -*- coding: utf-8 -*-
#
# Do the solver's internal waves follow the rays? The viscous core flow of the Love-number solver in the
# meridional plane against the Hamiltonian rays (P1 v2, Sect. 3 and Fig. rays_buoyancy).
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
# 03/11/2026 Machiel Bos
#===============================================================================
"""The ray tracer (rays_hamiltonian.py) and the solver are independent: the tracer integrates the local
dispersion relation fh^2 = (xi.s)^2 + n^2 (xi.r)^2 in a sphere, the solver the full viscous, compressible,
self-gravitating equations of flattened PREM (no ray approximation). In the solver the waves show up as
shear layers: short-wavelength flow on top of the smooth tidal flow. This script

  - solves the viscous core (delta, core K, band, mantle and inner core K 6 / N 24, core N 30, as Fig. 5) for
    PREM and for a neutrally stratified core (N^2 = 0, convergence_stratification.stratified(0));
  - rebuilds the displacement in the meridional plane from the spherical-harmonic and Chebyshev
    coefficients of the core layers, keeping only degrees n >= NCUT (the shear layers; |u| is independent
    of longitude for one m);
  - traces the ray for the same frequency (PREM N^2, or N = 0 for the neutral core);
  - measures: E_ray = share of the short-wavelength energy within DIST of the last 60 ray segments, against
    the share of the core area there (area of revolution, weight s); E_none = share in the region where no
    wave of that frequency can travel (Rays.disc < 0), against its area share. Rays in the solver: E_ray >
    area share on an attractor, E_none < area share.
Columns 'int': the same shares with the 20 km next to each wall left out (the Ekman layers carry short-wavelength
flow of their own, also inside a no-wave region).
Usage: python scripts/core_flow_rays.py KCORE DELTA_KM BAND [F ...] [--workers W] [--tag TAG]
  ->  results/core_flow_rays_K<K>_d<delta>[_TAG].npz, .txt
(memory: K72 band 12 ~4.8 GB per solve: W x peak within the budget; one case = one ray + one solve)"""
import os
import sys
import math
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.basis import ybar
import rays_hamiltonian as rh

NCUT = 20
DIST = 0.015        # r_cmb
NREF = 2000
ICB, CMB = 1221.5e3, 3480.0e3
NTH = 361
WALL = 20e3          # [m] left out next to each wall in the 'int' shares


def solve(f, Kc, delta, band, neutral):
    m = 1 if f < 1.5 else 2
    if neutral:
        from convergence_stratification import stratified
        md = stratified(0.0)
    else:
        md = EarthModel('data/PREM.dat')
    md = md.viscous_core(f * OMEGA_EARTH, delta=delta)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': 30}]})
    S = TidalSolver(md, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=f * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', band=band)
    S.solve()
    return S, md, m


def flow(S, md, m, th, nr=160):
    """|u| of degrees >= NCUT on (r, theta) in the core layers; returns r, |u_hp|^2 (nr per layer)."""
    lays = [i for i in range(md.n_layers) if md.r0[i] >= ICB - 1.0 and md.r1[i] <= CMB + 1.0]
    R, E = [], []
    Y = {}
    for i in lays:
        x = np.linspace(-1, 1, nr)
        s = md.r0[i] + (x + 1) * (md.r1[i] - md.r0[i]) / 2
        T = S._radial(i, x)[0][..., 0]                        # (N, nr)
        ur = np.zeros((nr, len(th)), complex)
        ut, up = ur.copy(), ur.copy()
        for j, (typ, n) in enumerate(S.B[i].fields):
            if typ == 'P' or n < NCUT:
                continue
            if n not in Y:
                Y[n] = (ybar(n, m, th, 1)[0], ybar(n, m, th, 1)[1], m * ybar(n, m, th, 0, over_sin=True)[0])
            y0, y1, ys = Y[n]
            a = (S.layer_coef(i, j)[:, None] * T).sum(0)[:, None]
            if typ == 'U':
                ur += a * y0
            elif typ == 'V':
                ut, up = ut + a * y1, up - a * ys
            else:
                ut, up = ut - a * ys, up + a * y1
        R.append(s)
        E.append(abs(ur) ** 2 + abs(ut) ** 2 + abs(up) ** 2)
    r = np.concatenate(R)
    o = np.argsort(r)
    return r[o], np.vstack(E)[o]


def ray(f, neutral):
    rh.H = 1e-3
    x, n2, dn2, rc = rh.buoyancy(EarthModel('data/PREM.dat'))
    if neutral:
        n2, dn2 = 0 * n2, 0 * dn2
    R = rh.Rays(x, n2, dn2, f / 2, rh.ETA)
    P, lk, path, werr, dmin = R.trace(NREF, NREF - 60)
    return R, path, rh.growth(lk)


def shares(R, path, r, th, E):
    """(energy share near the ray, area share near it, energy share where no waves, area share there)."""
    rr, tt = np.meshgrid(r / CMB, th, indexing='ij')
    s, z = rr * np.sin(tt), rr * np.cos(tt)
    # area of revolution on the (r, theta) grid: s dr dtheta r
    dr = np.gradient(r / CMB)[:, None]
    w = s * rr * dr * (th[1] - th[0])
    from scipy.spatial import cKDTree
    pts = np.column_stack([np.concatenate([path[:, 0], -path[:, 0]]), np.concatenate([path[:, 1], path[:, 1]])])
    d, _ = cKDTree(pts).query(np.column_stack([s.ravel(), z.ravel()]))
    near = (d.reshape(s.shape) < DIST)
    D = np.vectorize(lambda a, b: R.disc(abs(a), b) if rh.ETA < math.hypot(a, b) < 1 else 1.0)(s, z)
    none = D < 0
    out = []
    for keep in (np.ones_like(near), (rr > (ICB + WALL) / CMB) & (rr < 1 - WALL / CMB)):
        We, Wa = (E * w)[keep].sum(), w[keep].sum()
        out += [(E * w)[near & keep].sum() / We, w[near & keep].sum() / Wa,
                (E * w)[none & keep].sum() / We, w[none & keep].sum() / Wa]
    return out


def case(args):
    f, neutral, Kc, delta, band, th = args
    from earthtides.iers import iers_fit
    R, path, gr = ray(f, neutral)
    S, md, m = solve(f, Kc, delta, band, neutral)
    h2 = iers_fit(S, ClairautGeometry(md))['h2'].real
    r, E = flow(S, md, m, th)
    del S
    return f, neutral, gr, shares(R, path, r, th, E), h2, r, E, path


def main():
    from provenance import Run
    argv = sys.argv[1:]
    nproc, tag = 1, ''
    if '--workers' in argv:
        i = argv.index('--workers'); nproc = int(argv[i + 1]); del argv[i:i + 2]
    if '--tag' in argv:
        i = argv.index('--tag'); tag = '_' + argv[i + 1]; del argv[i:i + 2]
    Kc, dkm, band = int(argv[0]), float(argv[1]), int(argv[2])
    freqs = [float(a) for a in argv[3:]] or [0.907, 0.9175, 1.93227]
    th = np.linspace(1e-3, np.pi - 1e-3, NTH)
    out = f'results/core_flow_rays_K{Kc}_d{dkm:g}{tag}'
    with Run('solver core flow (degrees >= NCUT) against the Hamiltonian rays, PREM and neutral core',
             params=dict(K=Kc, delta_km=dkm, band=band, freqs=freqs, ncut=NCUT, dist=DIST, nref=NREF, wall=WALL,
                         workers=nproc),
             inputs=['data/PREM.dat']) as run_:
        lines = [f'viscous core delta {dkm:g} km, core K {Kc} N 30, band {band}; flow of degrees >= {NCUT}; '
                 f'near the ray = within {DIST} r_cmb of its last 60 segments ({NREF} reflections); int = without '
                 f'{WALL / 1e3:g} km next to the walls',
                 f'{"f":>8s} {"core":>8s} {"growth":>7s} {"E near":>7s} {"A near":>7s} {"E none":>7s} '
                 f'{"A none":>7s} {"int: E near":>11s} {"A near":>7s} {"E none":>7s} {"A none":>7s} {"h2":>14s}']
        arrs = dict(th=th[::2])
        jobs = [(f, nt, Kc, dkm * 1e3, band, th) for f in freqs for nt in (False, True)]
        with Pool(nproc, maxtasksperchild=1) as pool:
            for f, neutral, gr, sh, h2, r, E, path in pool.imap(case, jobs):
                tg = 'neutral' if neutral else 'PREM'
                lines.append(f'{f:8.5f} {tg:>8s} {gr:+7.4f} ' + ' '.join(f'{v:7.3f}' for v in sh[:4])
                             + f' {sh[4]:11.3f} ' + ' '.join(f'{v:7.3f}' for v in sh[5:]) + f' {h2:+14.10f}')
                print(lines[-1], flush=True)
                # saved maps: every second point, float32 (the shares above use the full grid)
                arrs[f'r_{tg}_{f}'], arrs[f'E_{tg}_{f}'] = r[::2], E[::2, ::2].astype(np.float32)
                arrs[f'path_{tg}_{f}'] = path
        run_.savez(out + '.npz', **arrs)
        txt = '\n'.join(lines)
        run_.write_text(out + '.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
