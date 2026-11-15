# -*- coding: utf-8 -*-
#
# Viscous limit against slipping core on Hough's Earth: the resonant shares, like for like with PREM (P1 v2 Sect. 5.3)
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
# 11/11/2026 Machiel Bos
#===============================================================================
"""For PREM without inner core the resonant share of h(2) is +0.0015 for the viscous core (delta 250, 125 m, K 48
and 72) and +0.0040 for the inviscid slipping core, with a stratified and with a neutral core
(share_vs_delta.py, strat_share_test.py). On Hough's Earth the two agreed in U4/U2 at one frequency, 1.002 cpsd
(wall_pressure_ingredients.py), which is not the same measure. This script computes the resonant shares on Hough's
Earth (examples/ex5: homogeneous Maclaurin spheroid, fluid core to 3480 km, exact similar spheroids) with the same
fit as for PREM (cubic background + FCN pole + 1/(s - 1)), over a sweep centred on Hough's FCN (1.0046 cpsd), for
  soft       mantle Vs 6 km/s, core Vp 1000 km/s (nearly incompressible)
  soft+comp  mantle Vs 6 km/s, core Vp 8 km/s
  soft+vp16, soft+vp32  the same with core Vp 16, 32 km/s (less top-heavy: N^2 = -(g / Vp)^2)
inviscid core K 6 and 10 (N 30) and viscous core at delta 250 and 125 m (K 48, N 30, band 12).
Result (11-11): soft agrees, +0.0020 both. PREM disagrees with a stratified, neutral and uniform-density core
(strat_share_test.py), so the core's rho(r) is not the discriminant. Two more variants separate what Hough's
Earth has and PREM has not (flattening constant with depth, no density jump at the CMB):
  clairaut   soft, but ClairautGeometry (first-order figure; homogeneous -> eps constant): the geometry class
  contrast   soft, ClairautGeometry, rho core 10635 / mantle 4500 kg/m3 (same mass): eps varies with depth
Result (11-11): the viscous share is the same under both geometries (0.0020); the slipping share triples under
the Clairaut map (0.0060). Two variants separate the value of eps from the form of the map:
  cl_exacteps  Clairaut map with eps scaled to the exact Maclaurin value (4.3276e-3 instead of 4.3236e-3)
  ex_cleps     exact similar spheroids with Clairaut's first-order eps (4.3236e-3)
Result (11-11): the form of the map decides (slipping 0.0060 under the Clairaut map with either eps, 0.0020 under
the exact spheroids with either eps; viscous 0.0020 in all four). The maps differ at O(eps^2), with a degree-4
part of the wall ~ eps^2. The direct test: exact spheroids with a degree-4 bump, R = s f(theta) (1 + b P4),
b = k eps^2 on every level surface:
  bump-1, bump+1, bump+3   k = -1, +1, +3
If the slipping share moves by O(eps) and the viscous one does not, the amplification (~1/eps) is shown directly.
--tag names the output (results/hough_share_test_<tag>.txt); without it the file of the first runs is overwritten.
Each variant's sweep is centred on its own FCN, found first by a coarse scan with the inviscid K 6 core.
Usage: python scripts/hough_share_test.py NPROC [--variants soft,...,bump+1] [--tag NAME] [--deltas 250,125] [--K 48]
                                            [--band 12] [--solver block|layers|sparse] [--slip-term]
       (--deltas none: inviscid runs only; --slip-term: the slipping wall with the term for a wall that is not a
        level surface, SlipTermSolver; uniform density only)
       ->  results/hough_share_test.txt (+ .npz)"""
import importlib.util
import sys
import tempfile
import warnings
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import least_squares

from earthtides.geometry import ExactSpheroidGeometry, ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
from fcn_resonance_extrapolation import KEYS
from slip_term import SlipTermSolver            # --slip-term: the slipping wall off a level surface

_spec = importlib.util.spec_from_file_location('hough1895', 'examples/ex5/hough1895.py')
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


NPROC = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
RHO = H.RHO                                     # 5500; the contrast keeps the mass: rho_m + 1000 (A / RC)^3
ALL = {'soft': (6.0, 1000.0, 'exact', RHO, RHO), 'soft+comp': (6.0, 8.0, 'exact', RHO, RHO),
       'clairaut': (6.0, 1000.0, 'clairaut', RHO, RHO),
       'contrast': (6.0, 1000.0, 'clairaut', 4500.0 + 1000.0 * (H.A / H.RC) ** 3, 4500.0),
       'cl_exacteps': (6.0, 1000.0, 'clairaut_exact_eps', RHO, RHO),
       'ex_cleps': (6.0, 1000.0, 'exact_clairaut_eps', RHO, RHO),
       'bump-1': (6.0, 1000.0, ('bump', -1), RHO, RHO), 'bump+1': (6.0, 1000.0, ('bump', 1), RHO, RHO),
       'bump+3': (6.0, 1000.0, ('bump', 3), RHO, RHO), 'bump+10': (6.0, 1000.0, ('bump', 10), RHO, RHO),
       # the same height (3 eps^2) at degree 6 and 8: steeper slopes (wall_fields.py, 2026-11-14)
       'bump6+3': (6.0, 1000.0, ('bump', 3, 6), RHO, RHO), 'bump8+3': (6.0, 1000.0, ('bump', 3, 8), RHO, RHO),
       # compressible core (Vp 8 km/s, as soft+comp): the Vp 1000 km/s core has a scaled system of condition
       # ~8e15 (round-off 3e-5 in h0 next to the FCN; lab notes 2026-11-12)
       'bump-1+comp': (6.0, 8.0, ('bump', -1), RHO, RHO), 'bump+1+comp': (6.0, 8.0, ('bump', 1), RHO, RHO),
       'bump+3+comp': (6.0, 8.0, ('bump', 3), RHO, RHO),
       # a compressible core of uniform density is top-heavy: N^2 = -(g / Vp)^2 = -84 Omega^2 at the CMB for
       # Vp 8 km/s. Stiffer cores scale it down: Vp 16: -21, Vp 32: -5 (PREM's core: +0.9 .. -5.6), 1000: -0.005
       'soft+vp16': (6.0, 16.0, 'exact', RHO, RHO), 'soft+vp32': (6.0, 32.0, 'exact', RHO, RHO),
       'bump+3+vp16': (6.0, 16.0, ('bump', 3), RHO, RHO), 'bump+3+vp32': (6.0, 32.0, ('bump', 3), RHO, RHO)}
VARIANTS = tuple(_opt('--variants', ','.join(ALL)).split(','))
DELTAS = tuple(float(x) for x in _opt('--deltas', '250,125').split(',') if x not in ('', 'none'))
SLIPTERM = '--slip-term' in sys.argv            # inviscid runs with SlipTermSolver (below)
KV = int(_opt('--K', 48))
# --band B: angular band of the viscous runs (default 12; the bump needs 16: band 12 is off by 0.1-0.5 in h0
# next to the FCN, lab notes 2026-11-12). --solver: linear solver of the viscous runs (block | layers | sparse)
BAND = int(_opt('--band', 12))
SOLVER = _opt('--solver', 'block')
TAG = _opt('--tag', '')
OUT = 'results/hough_share_test' + (f'_{TAG}' if TAG else '')
INVISCID = ((6, 30), (10, 30))
FCN0 = 1.00456                                  # Hough's FCN, rigid shell (examples/ex5)
# the PREM sweep (fcn_resonance_noic) with its points near the FCN (1.0022) moved to the variant's FCN f:
# sweep('soft') is the sweep of the 11-11 run (centred on Hough's 1.0046) to 1e-4
FAR = (0.90, 0.927, 0.96, 0.98, 0.99454, 0.9965, 0.9978, 0.9990)
OFFSETS = (-36, -26, -16, -10, -6, -4, -2, -1, 1, 2, 4, 6, 9, 14, 24, 44, 74, 154, 684)
SCAN = tuple(np.round(1 + np.arange(1, 41) * 5e-4, 5))      # coarse scan 1.0005 .. 1.02 for the FCN


def sweep(f):
    return tuple(sorted(set(FAR) | {round(f + o * 1e-4, 5) for o in OFFSETS}))


TMP = Path(tempfile.gettempdir()) / 'hough_share_models'


def model(variant, write=False):
    """The model file is written once by the parent (write=True); the workers only read it."""
    from earthtides.model import EarthModel
    TMP.mkdir(exist_ok=True)
    vs, vp, _, rho_c, rho_m = ALL[variant]
    path = TMP / f'HOUGH_{variant.replace("+", "_")}.dat'
    if write or not path.exists():
        return H.make(vs_m=vs, vp_c=vp, path=path, rho_c=rho_c, rho_m=rho_m)
    return EarthModel(str(path))


class BumpedSpheroid(ExactSpheroidGeometry):
    """Exact similar spheroids with a degree-4 bump on every level surface: R = s f(theta) (1 + b P4(cos theta))."""

    def __init__(self, flattening, b, n=4):
        super().__init__(flattening)
        self.b = b
        self.n = n                                            # degree of the bump (4; 6, 8: wall_fields.py)

    def map_derivs(self, i, s, theta):
        R, R_s, R_ss, R_t, R_st, R_tt = super().map_derivs(i, s, theta)
        c, sn = np.cos(theta), np.sin(theta)
        P4 = (35 * c ** 4 - 30 * c * c + 3) / 8
        dP4 = (140 * c ** 3 - 60 * c) / 8                   # dP4/dc
        ddP4 = (420 * c * c - 60) / 8
        if self.n != 4:                                       # P_n and its derivatives with respect to c
            Pn = np.polynomial.legendre.Legendre.basis(self.n)
            P4, dP4, ddP4 = Pn(c), Pn.deriv(1)(c), Pn.deriv(2)(c)
        g = 1 + self.b * P4
        g1 = -self.b * dP4 * sn                               # d/dtheta
        g2 = self.b * (ddP4 * sn * sn - dP4 * c)
        return (R * g, R_s * g, R_ss * g, R_t * g + R * g1, R_st * g + R_s * g1, R_tt * g + 2 * R_t * g1 + R * g2)


def geometry(variant, md):
    kind = ALL[variant][2]
    if isinstance(kind, tuple):                               # ('bump', k): b = k eps^2
        fl = H.maclaurin_flattening(OMEGA_EARTH)
        g = BumpedSpheroid(fl, kind[1] * fl ** 2, *kind[2:])
        return g, _Eps(fl), fl
    if kind == 'exact':
        g = ExactSpheroidGeometry(H.maclaurin_flattening(OMEGA_EARTH))
        return g, _Eps(g.fl), g.fl
    g = ClairautGeometry(md)
    if kind == 'clairaut':
        return g, g, g.flattening_surface
    if kind == 'exact_clairaut_eps':
        g = ExactSpheroidGeometry(g.flattening_surface)
        return g, _Eps(g.fl), g.fl
    scale = H.maclaurin_flattening(OMEGA_EARTH) / g.flattening_surface     # clairaut_exact_eps
    g = ClairautGeometry(md, scale=scale)
    return g, g, g.flattening_surface * scale


class _Eps:
    """iers_fit takes the surface ellipticity only for its normalisation (a common factor, cancels in the shares);
    ExactSpheroidGeometry has no eps(): the flattening of the similar spheroids is used."""
    def __init__(self, fl):
        self.fl = fl

    def eps(self, i, s):
        return np.full_like(s, self.fl), 0 * s, 0 * s


def solve(job):
    variant, s, delta, Kc, Nc = job
    w = s * OMEGA_EARTH
    md = model(variant)
    if delta > 0:
        md = md.viscous_core(w, delta=delta)
    g, g_fit, fl = geometry(variant, md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 16, 'rules': [{'fluid': True, 'K': Kc, 'N': Nc}]})
    cls = SlipTermSolver if (SLIPTERM and delta == 0) else TidalSolver
    S = cls(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
            omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
            frame='mantle', band=BAND if delta > 0 else None, slip_nonlevel=SLIPTERM,
                    linear_solver=SOLVER if delta > 0 else 'block')
    r = S.solve()
    if delta > 0:
        assert not any(S.true_fluid), 'a viscous layer was treated as inviscid'
    f = iers_fit(S, g_fit)
    f['k0'] = r['kk'] * (1 + fl / 3) ** -5
    print(f'{variant:9s} s {s:.5f} delta {delta:5.0f} K {Kc}  h0 {f["h0"].real:+.6f}  {S._solver_used}', flush=True)
    return job, np.array([complex(f[k]) for k in KEYS])


def basis(s, p):
    x = s - 1.0
    return np.stack([x ** k for k in range(4)] + [1.0 / (s - p), 1.0 / x], axis=1).astype(complex)


def fit(s, L, p0):
    """As fcn_resonance_noic.fit (one FCN pole), started at p0."""
    def coefs(p):
        B = basis(s, p)
        return B, np.linalg.lstsq(B, L, rcond=None)[0]

    def res(q):
        B, c = coefs(q[0] + 1j * q[1])
        r = B @ c - L
        return np.concatenate([r.real.ravel(), r.imag.ravel()])
    sol = least_squares(res, [p0.real, p0.imag], x_scale=1e-4, xtol=1e-15, ftol=1e-15)
    p = sol.x[0] + 1j * sol.x[1]
    B, c = coefs(p)
    return p, c, np.abs(B @ c - L).max(axis=0)


def locate(pool):
    """The FCN of each variant from a coarse scan with the inviscid K 6 core (the pole of the one-pole fit,
    started at the largest jump of h0 between neighbouring points)."""
    jobs = [(v, x, 0.0, 6, 30) for v in VARIANTS for x in SCAN]
    R = dict(pool.map(solve, jobs, chunksize=1))
    s = np.array(SCAN)
    out = {}
    for v in VARIANTS:
        L = np.array([R[(v, x, 0.0, 6, 30)] for x in SCAN])
        j = np.argmax(np.abs(np.diff(L[:, 0])))
        p, c, mx = fit(s, L, complex(0.5 * (s[j] + s[j + 1]), 1e-9))
        out[v] = float(round(p.real, 5))
    return out


def main():
    from provenance import Run
    with Run("resonant shares, viscous vs inviscid core, Hough's Earth (like for like with PREM)",
             params=dict(variants=VARIANTS, deltas=DELTAS, slip_term=SLIPTERM, Kv=KV, band=BAND, solver=SOLVER, inviscid=INVISCID,
                         models=ALL, scan=SCAN,
                         far=FAR, offsets=OFFSETS), inputs=[]) as run:
        cases = [(v, 0.0, K, N) for v in VARIANTS for K, N in INVISCID] + \
                [(v, d, KV, 30) for v in VARIANTS for d in DELTAS]
        for v in VARIANTS:
            model(v, write=True)
        with Pool(NPROC, maxtasksperchild=4) as pool:
            FCN = locate(pool)
            print('FCN from the scan:', FCN, flush=True)
            jobs = [(v, x, d, K, N) for v, d, K, N in cases for x in sweep(FCN[v])]
            R = dict(pool.map(solve, jobs, chunksize=1))
        run.params['fcn_scan'] = FCN
        lines = ["Hough's Earth (examples/ex5), mantle Vs 6 km/s; one-pole fit (cubic + FCN + 1/(s-1)); resonant parts "
                 'relative to that of h0; the sweep of each variant centred on its FCN from a coarse scan (K 6)',
                 '  ' + '  '.join(f'{v}: geometry {ALL[v][2]}, rho core {ALL[v][3]:.0f} mantle {ALL[v][4]:.0f}, '
                                  f'scan FCN {FCN[v]:.5f}' for v in VARIANTS),
                 f'{"variant":10s} {"core":>8s} {"K":>3s} {"N":>3s} {"FCN cpsd":>9s} {"FCN sd":>7s} {"Q":>7s} '
                 + ' '.join(f'{k:>9s}' for k in KEYS[1:]) + '  max |resid| h0']
        arrays = {}
        for v, d, K, N in cases:
            s = np.array(sweep(FCN[v]))
            L = np.array([R[(v, x, d, K, N)] for x in s])
            arrays[f'L_{v}_d{d:g}_K{K}_N{N}'] = L
            arrays[f's_{v}'] = s
            p, c, mx = fit(s, L, complex(FCN[v], 4e-4 * d / 1e3 if d > 0 else 1e-9))
            A = c[4]
            kind = f'{d:4.0f} m' if d > 0 else 'inviscid'
            lines.append(f'{v:10s} {kind:>8s} {K:3d} {N:3d} {p.real:9.6f} {1 / (p.real - 1):7.1f} '
                         f'{1 / (2 * p.imag) if p.imag else 0:7.0f} '
                         + ' '.join(f'{(A[i] / A[0]).real:+9.5f}' for i in range(1, len(KEYS)))
                         + f'  {mx[0]:.1e}')
        run.write_text(OUT + '.txt', '\n'.join(lines) + '\n')
        run.savez(OUT + '.npz', **arrays)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
