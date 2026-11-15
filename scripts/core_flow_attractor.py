# -*- coding: utf-8 -*-
#
# Do the solver's internal waves collect on the ray attractor as the viscosity falls? The viscous core
# flow against the attractor loop of the Hamiltonian ray, for delta -> 0 at rising resolution (P1 v2, Sect. 3).
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
# 04/11/2026 Machiel Bos
#===============================================================================
"""Follow-up of core_flow_rays.py, which found no energy on the attractors at delta 1 km / core K 72. Two
changes:

  - the measure: the share of the short-wavelength energy (degrees >= NCUT) within DIST of the attractor
    LOOP only (the last p segments of the ray, p its period, found from the wall-reflection points),
    against the share of the core area there; ratio > 1 means energy collects on the attractor. The old
    measure (last 60 segments) covered 30-80 % of the core;
  - radial resolution: viscous_core leaves the core interior (1261-3440 km) as one Chebyshev layer, with
    N = 30 nodes ~110 km apart in its middle, too coarse for beams of tens of km whatever K. --split S
    cuts it into S equal sub-layers (the model is unchanged, split_layer), and an optional sixth field
    sets N in those sub-layers (the Ekman sub-layers keep N 30).

Each case is f:core:delta_km:K:S[:N] (core PREM, neutral (N^2 = 0) or layered (base); N default 30). Per case: h2, peak RSS of
its worker, solve time, period p, shares for DIST and DIST/2, with and without WALL next to the walls.
Usage: python scripts/core_flow_attractor.py CASE [CASE ...] [--workers W] [--tag TAG] [--after PID] [--band B] [--solid-P M]
                                             [--max-gb G]
  --after PID   wait until process PID has ended (chain behind a running job)
  --full-maps   save the maps at every grid point (default every second; the shares always use all)
  --band B      angular band (default 12; 8 away from the FCN saves ~25 %)
  --solid-P M   TidalSolver(solid_P=M): the potential in the solid layers at their own degrees + M
  --sphere      no flattening (ClairautGeometry scale 0) and no centrifugal force (omega_rot 0), Coriolis kept
                (CORE_FLOW_SCALE for the workers)
  --max-gb G    (one worker) each case in its own process with its address space capped at G GB (Linux): a
                case that runs out of memory gives a 'failed' line and the next case runs
  ->  results/core_flow_attractor[_TAG].npz, .txt
(memory: measure one case at a new K and S first; workers x peak within the budget)"""
import os
import sys
import time
import warnings
try:
    import resource                         # not on Windows: no memory cap, no peak RSS
except ImportError:
    resource = None
from contextlib import nullcontext
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
_argv, sys.argv = sys.argv, sys.argv[:1]     # rays_hamiltonian reads its step H from argv at import
import rays_hamiltonian as rh
import core_flow_rays as cfr
sys.argv = _argv

NCUT = cfr.NCUT
DIST = 0.015        # r_cmb
NREF = 2000
WALL = 20e3
BAND = 12           # default; --band B (CORE_FLOW_BAND for the workers)
PMAX = 40           # longest attractor period searched (segments)
PTOL = 2e-3         # r_cmb: reflection points that repeat within this close the loop
ROT = np.radians(20.0)  # control: the loop turned by this angle about the centre


def base(core):
    """The elastic model of a core: PREM, neutral (N^2 = 0) or layered (neutral with stable layers of 140 km
    below the CMB and 150 km above the ICB, N ~ Omega: convergence_stratification.layered)."""
    import convergence_stratification as cs
    return {'PREM': lambda: EarthModel('data/PREM.dat'), 'neutral': lambda: cs.stratified(0.0),
            'layered': cs.layered}[core]()


def model(f, delta, core, split):
    md = base(core)
    if core == 'layered':                            # one viscosity for the three core layers (mean density)
        liq = [i for i in range(md.n_layers) if abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1.0]
        rho_m = (md.mass(liq[-1], md.r1[liq[-1]]) - md.mass_below[liq[0]]) / (
            4 / 3 * np.pi * (md.r1[liq[-1]] ** 3 - md.r0[liq[0]] ** 3))
        md = md.viscous_core(f * OMEGA_EARTH, eta=rho_m * OMEGA_EARTH * delta ** 2)
    else:
        md = md.viscous_core(f * OMEGA_EARTH, delta=delta)
    if split > 1:
        i = max((j for j in range(md.n_layers) if cfr.ICB - 1 <= md.r0[j] and md.r1[j] <= cfr.CMB + 1),
                key=lambda j: md.r1[j] - md.r0[j])
        md = md.split_layer(i, list(np.linspace(md.r0[i], md.r1[i], split + 1)[1:-1]))
    return md


def solve(f, Kc, delta, core, split, Ni=30):
    m = 1 if f < 1.5 else 2
    md = model(f, delta, core, split)
    sc = float(os.environ.get('CORE_FLOW_SCALE', 1.0))     # --sphere: 0, and no centrifugal force
    g = ClairautGeometry(md, scale=sc)                     # (omega_rot scaled with the flattening, as paper_numbers)
    rules = [{'fluid': True, 'K': Kc, 'N': 30}]
    if Ni != 30:                                     # the interior (sub-)layers; Ekman sub-layers are <= 40 km
        rules.append({'fluid': True, 'thickness_km': [100, 1e5], 'N': Ni})
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': rules})
    S = TidalSolver(md, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=f * OMEGA_EARTH, omega_rot=OMEGA_EARTH * np.sqrt(sc), wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', band=int(os.environ.get('CORE_FLOW_BAND', BAND)),
                    solid_P=None if os.environ.get('CORE_FLOW_SOLID_P', '') == '' else int(os.environ['CORE_FLOW_SOLID_P']))
    S.solve()
    return S, md, m, g


def loop(f, core):
    """Ray, its growth, period p and the path of its last p segments (the attractor loop; the last PMAX
    segments if the ray has not closed)."""
    rh.H = 1e-3
    x, n2, dn2, rc = rh.buoyancy(base('PREM' if core == 'neutral' else core))
    if core == 'neutral':
        n2, dn2 = 0 * n2, 0 * dn2
    R = rh.Rays(x, n2, dn2, f / 2, rh.ETA)
    P, lk, path, werr, dmin = R.trace(NREF, NREF - PMAX - 2)
    p = next((k for k in range(1, PMAX + 1)
              if np.abs(P[-2 * k:] - P[-3 * k:-k]).max() < PTOL), None)
    r = np.hypot(*path.T)
    wall = np.flatnonzero((np.abs(r - 1) < 1e-7) | (np.abs(r - rh.ETA) < 1e-7))
    seg = path[wall[-(p or PMAX)]:]
    return R, seg, rh.growth(lk), p


def asym(seg):
    """Median distance (r_cmb) of the loop's mirror image from the loop: 0 for a symmetric attractor."""
    from scipy.spatial import cKDTree
    q = seg * np.where(seg[:, :1] < 0, [-1.0, 1.0], [1.0, 1.0])     # folded across the axis, as the flow map
    return float(np.median(cKDTree(q).query(q * [1.0, -1.0])[0]))


def _grid(r, th):
    rr, tt = np.meshgrid(r / cfr.CMB, th, indexing='ij')
    s, z = rr * np.sin(tt), rr * np.cos(tt)
    w = s * rr * np.gradient(r / cfr.CMB)[:, None] * (th[1] - th[0])
    keep = (rr > (cfr.ICB + WALL) / cfr.CMB) & (rr < 1 - WALL / cfr.CMB)
    return s, z, w, keep


def _dist(seg, s, z):
    """Distance of every grid point from the loop and its mirror in the equator."""
    from scipy.spatial import cKDTree
    # rays run in the whole meridional plane (s < 0 across the axis): fold s; and add the mirror in the
    # equator: the mirror of an attractor is an attractor, and the forced flow (one parity) carries both
    pts = seg * np.where(seg[:, :1] < 0, [-1.0, 1.0], [1.0, 1.0])
    pts = np.concatenate([pts, pts * [1.0, -1.0]])
    return cKDTree(pts).query(np.column_stack([s.ravel(), z.ravel()]))[0].reshape(s.shape)


def shares(R, seg, r, th, E):
    """(E, A) near the loop and its mirror for DIST and DIST/2, whole core and without WALL."""
    s, z, w, keep = _grid(r, th)
    d = _dist(seg, s, z)
    out = []
    for dist in (DIST, DIST / 2):
        for k in (np.ones_like(keep), keep):
            near = (d < dist) & k
            out += [(E * w)[near].sum() / (E * w)[k].sum(), w[near].sum() / w[k].sum()]
    return out


def rotated(seg, r, th, E):
    """Control: E/A (DIST/2, without WALL) on the loop turned by +-ROT about the centre (radii kept, so it
    still runs wall to wall, but no longer along the rays); mean of the two turns."""
    s, z, w, keep = _grid(r, th)
    out = []
    for a in (ROT, -ROT):
        c, sn = np.cos(a), np.sin(a)
        d = _dist(seg @ np.array([[c, sn], [-sn, c]]), s, z)
        near = (d < DIST / 2) & keep
        out.append(((E * w)[near].sum() / (E * w)[keep].sum()) / (w[near].sum() / w[keep].sum()))
    return float(np.mean(out))


def case(c):
    f, core, dkm, Kc, split, Ni = c
    gb = float(os.environ.get('CORE_FLOW_MAX_GB', 0))
    if gb and resource is not None and sys.platform.startswith('linux'):
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * 1e9), int(gb * 1e9)))
    from earthtides.iers import iers_fit
    R, seg, gr, p = loop(f, core)
    t0 = time.time()
    S, md, m, g = solve(f, Kc, dkm * 1e3, core, split, Ni)
    ts = time.time() - t0
    h2 = iers_fit(S, g)['h2'].real
    th = np.linspace(1e-3, np.pi - 1e-3, 8 * Kc + 1)
    r, E = cfr.flow(S, md, m, th, nr=max(120 if split > 1 else 160, 4 * Ni))
    del S
    rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6)
           if resource is not None else float('nan'))
    return c, gr, p, asym(seg), shares(R, seg, r, th, E) + [rotated(seg, r, th, E)], h2, rss, ts, r, th, E, seg


def parse(a):
    f, core, d, K, S, *Ni = a.split(':')
    if core not in ('PREM', 'neutral', 'layered') or len(Ni) > 1:
        raise ValueError(f'case is f:core:delta_km:K:S[:N], core PREM, neutral or layered: {a}')
    return float(f), core, float(d), int(K), int(S), int(Ni[0]) if Ni else 30


def isolated(cases):
    """Each case in its own one-worker executor: a worker killed by the memory cap breaks only its case."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    for c in cases:
        try:
            with ProcessPoolExecutor(1) as ex:
                yield ex.submit(case, c).result()
        except (MemoryError, BrokenProcessPool) as e:
            yield c, type(e).__name__


def wait_for(pid):
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:                      # exists, owned by another user
            pass
        time.sleep(60)


def main():
    from provenance import Run
    argv = sys.argv[1:]
    nproc, tag = 1, ''
    if '--workers' in argv:
        i = argv.index('--workers'); nproc = int(argv[i + 1]); del argv[i:i + 2]
    full_maps = '--full-maps' in argv
    if full_maps:
        argv.remove('--full-maps')
    if '--tag' in argv:
        i = argv.index('--tag'); tag = '_' + argv[i + 1]; del argv[i:i + 2]
    after = maxgb = None
    if '--after' in argv:
        i = argv.index('--after'); after = int(argv[i + 1]); del argv[i:i + 2]
    if '--max-gb' in argv:
        i = argv.index('--max-gb'); maxgb = float(argv[i + 1]); del argv[i:i + 2]
        if nproc != 1:
            raise ValueError('--max-gb runs one case at a time: --workers 1')
        os.environ['CORE_FLOW_MAX_GB'] = str(maxgb)
    band = int(argv.pop(argv.index('--band') + 1)) if '--band' in argv else BAND
    if '--band' in argv:
        argv.remove('--band')
    solid_P = int(argv.pop(argv.index('--solid-P') + 1)) if '--solid-P' in argv else None
    if '--solid-P' in argv:
        argv.remove('--solid-P')
    sphere = '--sphere' in argv
    if sphere:
        argv.remove('--sphere')
    os.environ['CORE_FLOW_SCALE'] = '0' if sphere else '1'
    os.environ['CORE_FLOW_BAND'] = str(band)
    os.environ['CORE_FLOW_SOLID_P'] = '' if solid_P is None else str(solid_P)
    cases = [parse(a) for a in argv]
    if after:
        print(f'waiting for process {after}', flush=True)
        wait_for(after)
    out = f'results/core_flow_attractor{tag}'
    with Run('solver core flow (degrees >= NCUT) on the ray attractor loop, delta -> 0 at rising K and '
             'radial split', params=dict(cases=[':'.join(map(str, c)) for c in cases], ncut=NCUT, dist=DIST,
                                         nref=NREF, wall=WALL, band=band, solid_P=solid_P, workers=nproc,
                                         max_gb=maxgb, full_maps=full_maps, sphere=sphere),
             inputs=['data/PREM.dat']) as run_:
        lines = [f'flow of degrees >= {NCUT} near the attractor loop (last p segments of {NREF} reflections) and '
                 f'its mirror in the equator (asym: median distance of the mirror from the loop, r_cmb); '
                 f'shares E (energy) and A (area) within {DIST} and {DIST / 2} r_cmb; int = without '
                 f'{WALL / 1e3:g} km next to the walls; rot = E2int/A on the loop turned by +-{np.degrees(ROT):g} '
                 f'deg about the centre (control); core K / N 30, band {band}, solid_P {solid_P}{", no flattening (sphere)" if sphere else ""}; S = core '
                 f'interior split, '
                 f'N its sub-layers',
                 f'{"f":>7s} {"core":>7s} {"delta":>5s} {"K":>4s} {"S":>3s} {"N":>3s} {"growth":>7s} {"p":>3s} '
                 f'{"asym":>6s} '
                 + ' '.join(f'{h:>6s}' for h in ('E1', 'A1', 'E1int', 'A1int', 'E2', 'A2', 'E2int', 'A2int'))
                 + f' {"E1int/A":>7s} {"E2int/A":>7s} {"rot":>7s} {"h2":>14s} {"GB":>5s} {"s":>6s}']
        arrs = {}
        with (nullcontext() if maxgb else Pool(nproc, maxtasksperchild=1)) as pool:
            for res in (isolated(cases) if maxgb else pool.imap(case, cases)):
                f, core, dkm, Kc, split, Ni = res[0]
                head = f'{f:7.4f} {core:>7s} {dkm:5.2f} {Kc:4d} {split:3d} {Ni:3d} '
                if len(res) == 2:
                    lines.append(head + f'failed: {res[1]} (memory cap {maxgb:g} GB)')
                    print(lines[-1], flush=True)
                    continue
                c, gr, p, am, sh, h2, rss, ts, r, th, E, seg = res
                lines.append(head + f'{gr:+7.4f} {p or 0:3d} {am:6.3f} '
                             + ' '.join(f'{v:6.3f}' for v in sh[:8])
                             + f' {sh[2] / sh[3]:7.2f} {sh[6] / sh[7]:7.2f} {sh[8]:7.2f} {h2:+14.10f} {rss:5.1f} '
                             f'{ts:6.0f}')
                print(lines[-1], flush=True)
                k = f'{core}_{f}_d{dkm:g}_K{Kc}_S{split}' + (f'_N{Ni}' if Ni != 30 else '')
                st = 1 if full_maps else 2           # saved maps: every point (--full-maps) or every second
                arrs[f'r_{k}'], arrs[f'th_{k}'] = r[::st], th[::st]
                arrs[f'E_{k}'], arrs[f'loop_{k}'] = E[::st, ::st].astype(np.float32), seg
        run_.savez(out + '.npz', **arrs)
        txt = '\n'.join(lines)
        run_.write_text(out + '.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
