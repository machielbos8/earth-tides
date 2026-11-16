# -*- coding: utf-8 -*-
#
# The solution itself next to the core-mantle boundary: slipping against viscous core, Hough's Earth.
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
# 14/11/2026 Machiel Bos
#===============================================================================
"""The ladders and share tests store Love numbers only. On a bumped wall the slipping core and the viscous core
give different latitude terms at every delta (lab notes 2026-11-12 .. 14), and the Love numbers do not say where
the two solutions differ. This script stores the fields on (radius, colatitude) in a window around the
core-mantle boundary (Machiel 2026-11-14), for the variants of hough_share_test.py, each with
  slip     inviscid slipping core (core K --Kslip, N 30, solver block; as hough_share_test)
  visc     viscous core, delta --delta m (core K --K, N --N, band --band, solver layers, pressure unknown)
at the frequencies --s (cpsd). Per case and frequency, in every layer that touches the window:
  r (true radius), u_r, u_theta, u_lambda, phi1, div u, rho1 = -div(rho u) (Eulerian), the six stress components,
  Pi (lambda div u), un = u . grad Phi0 / |grad Phi0| (displacement across the level surfaces) and
  pE = -(mean normal stress) + rho u . grad Phi0 (pressure at a fixed point in space); amplitudes per unit tidal potential, mantle frame, as the solver
  defines them (u = Re[(u_r, u_theta, -i u_lambda) e^(i(omega t + m lambda))]).
The text file gives the coefficients U_n, V_n (n = 2, 4, 6) and W_n (1, 3, 5) of the expansion and the degree-2,
4 and 6 parts of tau_rr and phi1 (on Ybar_n) at a few depths below and above the wall, divided by U_2 on the
mantle side of the wall (the amplitude of the resonance cancels), and the Love numbers. (u_theta is not
projected on dYbar_n/dtheta: the rigid rotation W_1 projects on every one of them.)
Usage: python scripts/wall_fields.py VARIANT[,VARIANT] [--s 0.96,1.00396] [--delta 250] [--K 36] [--N 40]
       [--band 16] [--Kslip 10] [--below 200] [--above 200] (km) [--nr 40] [--kinds slip,visc] [--level-reference] [--slip-term] [--tag NAME]
   ->  results/wall_fields_<variants>[_tag].{txt,npz}, figures by scripts/plot_wall_fields.py"""
import os
import resource
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool

_argv, sys.argv = sys.argv, sys.argv[:1]         # hough_share_test reads its own options from argv
import hough_share_test as hs
sys.argv = _argv
from earthtides.geometry import OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
from earthtides.basis import ybar
from earthtides.jets import _D
from provenance import Run

CMB = 3480.0e3
KEYS = ('h0', 'h2', 'l1', 'l2')
JET = ('v', 'r', 't', 'rr', 'rt', 'tt')
FIELDS = ('ur', 'ut', 'ul', 'phi', 'div', 'rho1', 'trr', 'ttt', 'tll', 'trt', 'trl', 'ttl', 'Pi')
# --level-reference: Phi0 = W(s) on the coordinate surfaces (every coordinate surface, the wall too, is then a
# level surface) instead of the potential that Poisson's equation gives for the density (refpot.py)
CONSISTENT = '--level-reference' not in sys.argv
DEPTHS = (-100e3, -20e3, -1.0, 1.0)              # m above the wall (negative: in the core); + multiples of delta


SlipTermSolver = hs.SlipTermSolver               # the slipping-wall condition with the term for a non-level wall


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def layer_fields(S, i, s):
    """All fields of layer i at the radii s (coordinate radius): dict of (len(s), n_theta) arrays."""
    md = S.model
    x = 2 * (s - md.r0[i]) / (md.r1[i] - md.r0[i]) - 1
    p = S.props(i, s)
    tot = [{k: 0.0 for k in JET} for _ in range(4)]
    q = {k: 0.0 for k in JET}
    for j, comps in enumerate(S._fields_at(i, x, p)):
        c = S.layer_coef(i, j)
        sh = (len(c),) + p.r.shape
        for a, d in enumerate(comps):
            for k in JET:
                tot[a][k] = tot[a][k] + np.tensordot(c, np.broadcast_to(d[k], sh), axes=1)
        if hasattr(comps, 'q'):
            for k in JET:
                q[k] = q[k] + np.tensordot(c, np.broadcast_to(comps.q[k], sh), axes=1)
    uA, uB, uC, ph = (_D(**t) for t in tot)
    mixed = bool(p.get('mixed'))
    e = S._stress(uA, uB, uC, p, q=_D(**q) if mixed else None)
    full = lambda f: np.broadcast_to(f, p.r.shape).astype(complex)
    # Eulerian density perturbation -div(rho u) = -(rho div u + u . grad rho); rho_t is d/dtheta at fixed r
    rho1 = -(p.rho * e.D.v + uA.v * p.rho_r + uB.v * p.rho_t / p.r)
    # displacement across the level surfaces: u . grad Phi0 (ph0.t is d/dtheta at fixed r), and the pressure at
    # a fixed point in space: -(mean normal stress) - u . grad p0, grad p0 = -rho grad Phi0
    ugp = uA.v * p.ph0['r'] + uB.v * p.ph0['t'] / p.r
    gn = np.sqrt(p.ph0['r'] ** 2 + (p.ph0['t'] / p.r) ** 2)
    pE = -(e.trr.v + e.ttt.v + e.tpp.v) / 3 + p.rho * ugp
    out = dict(r=np.broadcast_to(p.r, p.r.shape).astype(float), un=full(ugp / gn), pE=full(pE), ur=full(uA.v), ut=full(uB.v), ul=full(uC.v),
               phi=full(ph.v), div=full(e.D.v), rho1=full(rho1), trr=full(e.trr.v), ttt=full(e.ttt.v),
               tll=full(e.tpp.v), trt=full(e.trt.v), trl=full(e.trp.v), ttl=full(e.ttp.v),
               Pi=full(q['v']) if mixed else full(p.lam * e.D.v))
    # the expansion itself: radial profile of every spheroidal (U, V) and toroidal (W) coefficient
    T = S._radial(i, x)[0][..., 0]                                   # (N, len(s))
    for typ in 'UVW':
        js = [(n, j) for j, (t, n) in enumerate(S.B[i].fields) if t == typ]
        out['deg' + typ] = np.array([n for n, _ in js])
        out['c' + typ] = np.array([S.layer_coef(i, j) @ T for _, j in js]).reshape(len(js), len(s))
    return out


def solve(job):
    """job = (variant, kind, s, delta, K, N, band, below, above, nr) -> fields in the window, Love numbers."""
    variant, kind, s, delta, K, N, band, below, above, nr = job
    w = s * OMEGA_EARTH
    md = hs.model(variant)
    if kind == 'visc':
        md = md.viscous_core(w, delta=delta)
    g, g_fit, fl = hs.geometry(variant, md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 16, 'rules': [{'fluid': True, 'K': K, 'N': N}]})
    S = (SlipTermSolver if '--slip-term' in sys.argv else TidalSolver)(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=CONSISTENT,
                    frame='mantle', band=band if kind == 'visc' else None,
                    linear_solver='layers' if kind == 'visc' else 'block', fluid_pressure=(kind == 'visc'),
                    slip_nonlevel='--slip-term' in sys.argv)
    S.solve()
    fit = iers_fit(S, g_fit)
    L = np.array([complex(fit[k]) for k in KEYS])
    lo, hi = CMB - below, CMB + above
    out = dict(theta=S.theta[0, 0].copy(), w=np.ravel(S.w).copy(), love=L)
    lays = []
    for i in range(md.n_layers):
        a, b = max(md.r0[i], lo), min(md.r1[i], hi)
        if b - a < 1e-3:
            continue
        # Chebyshev-like spacing: dense at both ends of the part of the layer inside the window
        s_ = a + (b - a) * (1 - np.cos(np.pi * np.arange(nr) / (nr - 1))) / 2
        F = layer_fields(S, i, s_)
        out[f's_{len(lays)}'] = s_
        for k, v in F.items():
            out[f'{k}_{len(lays)}'] = v
        lays.append((md.r0[i], md.r1[i], bool(S.fluid_layers()[i])))
    out['layers'] = np.array(lays)
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6)
    print(f'{variant:8s} {kind:4s} s {s:.5f}: h0 {L[0].real:+.6f} h2 {L[1].real:+.3e}, {len(lays)} layers in the '
          f'window, peak {gb:.1f} GB, {S._solver_used}', flush=True)
    return out


def part(d, name, s_m, n, deriv=False):
    """Degree-n part of field name at coordinate radius s_m (nearest stored radius on the right side)."""
    th, w = d['theta'], d['w']
    best = None
    for k in range(len(d['layers'])):
        s_ = d[f's_{k}']
        if s_[0] - 1e-6 <= s_m <= s_[-1] + 1e-6:
            j = int(np.argmin(abs(s_ - s_m)))
            best = d[f'{name}_{k}'][j]
            # a point on a boundary belongs to two layers: below the wall take the lower, above the upper
            if s_m < CMB:
                break
    Y = ybar(n, 1, th, 1)
    b = Y[1] if deriv else Y[0]
    return complex(np.sum(w * best * b) / np.sum(w * b * b))


def coef(d, typ, s_m, n):
    """Coefficient (typ, n) of the expansion at coordinate radius s_m; 0 when the layer does not carry it."""
    val = 0.0
    for k in range(len(d['layers'])):
        s_ = d[f's_{k}']
        if s_[0] - 1e-6 <= s_m <= s_[-1] + 1e-6:
            deg = list(d[f'deg{typ}_{k}'])
            val = complex(d[f'c{typ}_{k}'][deg.index(n), int(np.argmin(abs(s_ - s_m)))]) if n in deg else 0.0
            if s_m < CMB:
                break
    return val


def main():
    variants = sys.argv[1].split(',')
    freqs = [float(x) for x in _opt('--s', '0.96,1.00396').split(',')]
    delta, K, N = float(_opt('--delta', 250)), int(_opt('--K', 36)), int(_opt('--N', 40))
    band, Ks = int(_opt('--band', 16)), int(_opt('--Kslip', 10))
    below, above, nr = float(_opt('--below', 200)) * 1e3, float(_opt('--above', 200)) * 1e3, int(_opt('--nr', 40))
    tag = _opt('--tag', '')
    out = 'results/wall_fields_' + '_'.join(v.replace('+', '_') for v in variants) + (f'_{tag}' if tag else '')
    with Run('fields next to the core-mantle boundary, slipping against viscous core, Hough\'s Earth',
             params=dict(variants=variants, s=freqs, delta=delta, K=K, N=N, band=band, Kslip=Ks, below=below,
                         above=above, nr=nr, consistent_reference=CONSISTENT, slip_term='--slip-term' in sys.argv), inputs=[]) as run:
        for v in variants:
            hs.model(v, write=True)
        kinds = _opt('--kinds', 'slip,visc').split(',')
        jobs, R = [], []
        if 'slip' in kinds:
            jobs = [(v, 'slip', s, 0.0, Ks, 30, None, below, above, nr) for v in variants for s in freqs]
            with Pool(2, maxtasksperchild=1) as pool:
                R = pool.map(solve, jobs, chunksize=1)
        if 'visc' in kinds:
            vj = [(v, 'visc', s, delta, K, N, band, below, above, nr) for v in variants for s in freqs]
            with Pool(1, maxtasksperchild=1) as pool:                  # one heavy solve at a time
                R += pool.map(solve, vj, chunksize=1)
            jobs += vj
        save, lines = {}, [f'Hough\'s Earth, fields around the CMB ({below / 1e3:g} km below .. {above / 1e3:g} km '
                           f'above); slip: core K {Ks} N 30; visc: delta {delta:g} m, core K {K} N {N} band {band}, '
                           'pressure unknown; all divided by U2 on the mantle side '
                           'of the wall (real, imaginary)']
        depths = sorted(set(DEPTHS) | {-30 * delta, -5 * delta, -delta})
        for job, d in zip(jobs, R):
            key = f'{job[0]}|{job[1]}|{job[2]:.5f}'
            for k, v in d.items():
                save[f'{key}|{k}'] = v
        for s in freqs:
            for v in variants:
                lines += ['', f'{v}, {s:.5f} cpsd']
                for kind in kinds:
                    d = R[jobs.index(next(j for j in jobs if j[:3] == (v, kind, s)))]
                    ref = coef(d, 'U', CMB + 1.0, 2)
                    lines.append(f'  {kind}: Love numbers ' + ' '.join(f'{x.real:+.6e}' for x in d['love'])
                                 + f'; U2 on the mantle side of the wall {ref.real:+.4e} {ref.imag:+.4e}')
                    cols = [('U', 2), ('U', 4), ('U', 6), ('U', 8), ('W', 1), ('W', 3), ('trr', 2), ('trr', 4),
                            ('trr', 6), ('trr', 8), ('un', 4), ('un', 6), ('un', 8), ('pE', 2), ('pE', 4),
                            ('pE', 6), ('pE', 8), ('phi', 4)]
                    lines.append(f'    {"above wall m":>12s} ' + ' '.join(f'{q + str(n):>21s}' for q, n in cols))
                    for z in depths:
                        row = []
                        for q, n in cols:
                            c = (coef(d, q, CMB + z, n) if len(q) == 1 else part(d, q, CMB + z, n)) / ref
                            row.append(f'{c.real:+10.3e}{c.imag:+10.2e}i')
                        lines.append(f'    {z:12.0f} ' + ' '.join(row))
        run.savez(out + '.npz', **save)
        run.write_text(out + '.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
