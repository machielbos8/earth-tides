# -*- coding: utf-8 -*-
#
# Love numbers of a flattened, rotating, anelastic Earth for a list of tidal
# waves, written to a JSON table (the input of etide).
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""Love numbers of a flattened, rotating, anelastic Earth for a list of tidal waves, written to a
JSON table (the input of etide).

For every wave (degree n, order m, frequency f) the solver is run once and its surface field is
reduced to the IERS parametrization h(0), h(2), l(0), l(1), l(2), h', l' (paper Sect. 3.9). Every
choice behind that parametrization is an option, so that the effect of each convention can be
computed (paper Sect. 5.1, Table 3):

  latitude    geocentric | geodetic      latitude in the basis functions
  direction   radial | normal            components along the geocentric radius or the normal
  surface     deformed | mean            displacement on the flattened surface or the mean sphere
  norm        equatorial | mean          tidal potential normalized at R_e or at the mean radius
  frame       mantle | surface           rigid rotation removed: mantle (Tisserand) or best-fit surface
  centrifugal on | off                   centrifugal force in the reference gravity

The IERS conventions are geocentric, radial, deformed, equatorial, mantle (and centrifugal on);
etide accepts only tables in these conventions. The physics options are the Earth model,
anelasticity (absorption band), rotation (Coriolis and inertia), the hydrostatic flattening, and
the resolution K, N.

Special waves: below `f_dyn` (default 0.005 cpd: 18.6 yr and longer) the dynamics of the fluid core
is solved at f_dyn, because a static fluid core is not uniquely defined (Longman 1963), with the
anelasticity at the wave's own frequency; the permanent tide (f = 0) gets the IERS nominal values,
so that positions stay conventional tide free. Degrees 3 and 4: one spherical solution per degree
at the M2 frequency, applied to every wave of that degree.

Wave groups (default): as in tidal-gravity practice (ETERNA/PREDICT), one Love number per wave
group, computed at the frequency of the group's largest wave and assigned to every wave of the
group. The default set (data/groups_default.txt, tools/design_groups.py) was designed from the
admittance of PREM and of the IERS model so that the displacement differs from the per-wave result
by less than 0.03 mm anywhere (tools/verify_groups.py); it needs 21 solves (seconds). --all-waves
gives every wave of the catalogue its own Love number (837 solves, minutes).

Set-up file: every physics option, the non-hydrostatic Earth (earthtides.nonhydro: excess J2 and CMB
flattening from supported mantle loads) and the run can be given in one TOML file;
earthtides-love --template prints an annotated example, earthtides-love --setup FILE runs it.

Command line: earthtides-love [--groups default|eterna|eterna-split|FILE | --all-waves] -o love.json
              earthtides-love --setup run.toml
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
FORMAT = 'earthtides-love'
FORMAT_VERSION = 1
IERS_CONVENTIONS = dict(latitude='geocentric', direction='radial', surface='deformed',
                        norm='equatorial', frame='mantle')
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'hp', 'lp', 'k')
F_M2 = 1.9322736 / 1.002737909      # cycles per day, M2 (for degrees 3 and 4)
F_TILT = 7.292115e-5 * 86400 / (2 * np.pi)    # [cpd] the tilt-over frequency, sigma = 1 (= K1)
TILT_WINDOW = 1e-5
IERS_PERMANENT = dict(h0=0.6078, h2=-0.0006, l0=0.0847, l2=0.0002)


@dataclass
class Options:
    """Physics, numerics and conventions of a Love-number table (stored in its metadata)."""
    model: str = 'PREM'
    anelastic: bool = True
    alpha: float = 0.15
    tm: float = 200.0
    # seismic reference period of the elastic model [s]: constant Q from t0 to tm, power law beyond. t0 = tm drops
    # the constant-Q band, the assumption behind the IERS anelastic values (Mathews et al. 1997: PREM at 1 s,
    # power law from 200 s); default 1 s (PREM)
    t0: float = 1.0
    dispersion: str = 'cot'
    rotation: bool = True
    flattening: bool = True
    # figure of the flattened Earth (geometry.figure): 'first-order' (Clairaut, eq. (map) of paper 1) or
    # 'level' (level surfaces of the reference potential: second order in the flattening and beyond)
    figure: str = 'level'       # default since 2026-11-15; every earlier table: 'first-order'
    centrifugal: bool = True
    frame: str = 'mantle'
    latitude: str = 'geocentric'
    direction: str = 'radial'
    surface: str = 'deformed'
    norm: str = 'equatorial'
    K: int = 4
    N: int = 24
    fluidcore_N: int = None     # Chebyshev polynomials per (weak-)fluid layer; None = N
    fluidcore_K: int = None     # degrees of U, V, W per (weak-)fluid layer; None = K
    # core_median: N values of the liquid core, e.g. (16, 20, 24, 30). An inviscid core in a shell has a
    # continuous spectrum, and any truncation puts pseudo-modes at frequencies that depend on the
    # resolution; a wave that falls on one gets a spurious value (h(2) off by up to 1e-3). Each wave is
    # then solved once per N and every parameter is the median (real and imaginary parts separately);
    # 'core_spread' records the largest deviation from it. None: one solve with fluidcore_N.
    core_median: tuple = None
    # K and N per layer from a rule set (solver.resolution_per_layer; the JSON file's content is
    # stored, so the table's metadata reproduces it); overrides K, N, fluidcore_K, fluidcore_N
    resolution: dict = None
    f_dyn: float = 0.005
    # non-hydrostatic Earth (earthtides.nonhydro): None, or a table with support, shells_km,
    # targets (J2, surface, cmb, e_f) and optionally sigma, prior; model_static: the same model with a
    # weak-solid core for the static support ('' = PREMs for PREM)
    nonhydrostatic: dict = None
    model_static: str = ''
    conventions: dict = field(init=False)

    def __post_init__(self):
        for key, allowed in (('dispersion', ('cot', 'astyanax')), ('figure', ('first-order', 'level')),
                             ('frame', ('mantle', 'surface')), ('latitude', ('geocentric', 'geodetic')),
                             ('direction', ('radial', 'normal')), ('surface', ('deformed', 'mean')),
                             ('norm', ('equatorial', 'mean'))):
            if getattr(self, key) not in allowed:
                raise ValueError(f'{key} must be one of {allowed}')
        if self.core_median is not None:
            if self.resolution is not None:
                raise ValueError('core_median sets the N of the core; it cannot be combined with resolution')
            self.core_median = tuple(int(x) for x in self.core_median)
        self.conventions = {k: getattr(self, k) for k in IERS_CONVENTIONS}
        if self.nonhydrostatic is not None:
            if not (self.flattening and self.centrifugal):
                raise ValueError('nonhydrostatic needs flattening and the centrifugal reference')
            self.nonhydrostatic = dict(self.nonhydrostatic)

    def is_iers(self):
        return self.conventions == IERS_CONVENTIONS and self.centrifugal


def model_path(name):
    """An Earth-model file: a path, or a name of the package data (PREM, PREMs, HOMOG)."""
    p = Path(name)
    if p.is_file():
        return p
    q = DATA / (name if name.endswith('.dat') else name + '.dat')
    if q.is_file():
        return q
    raise FileNotFoundError(f'Earth model {name}: not a file and not in {DATA}')


# ---------------------------------------------------------------------------- one wave
_CACHE = {}


def _model(opts):
    """(model, geometry, non-hydrostatic state or None), cached per process: the non-hydrostatic
    state is static, so one state serves every wave (and every anelastic tidal model)."""
    from .model import EarthModel
    from .geometry import figure, OMEGA_EARTH
    nh = opts.nonhydrostatic
    key = (opts.model, opts.flattening, opts.figure, json.dumps(nh, sort_keys=True), opts.model_static)
    if key not in _CACHE:
        md = EarthModel(str(model_path(opts.model)))
        geo = figure(md, opts.figure) if opts.flattening else None
        state = None
        if nh is not None:
            from .nonhydro import NonHydrostatic
            name = (nh.get('model_static') or opts.model_static
                    or ('PREMs' if Path(opts.model).stem == 'PREM' else ''))
            if not name:
                raise ValueError('nonhydrostatic: give model_static (the model with a weak-solid core)')
            state = NonHydrostatic(md, nh, geo, OMEGA_EARTH, EarthModel(str(model_path(name))))
            geo = state.geometry
        _CACHE[key] = (md, geo, state)
    return _CACHE[key]


def solve_wave(opts, m, f_cpd):
    """Solve the degree-2 tide of order m and frequency f_cpd [cycles per day]."""
    from .geometry import OMEGA_EARTH
    from .solver import TidalSolver, resolution_per_layer
    md0, geo, nh = _model(opts)
    w = max(f_cpd, opts.f_dyn) * 2 * np.pi / 86400.0
    w_an = max(f_cpd, 1 / 6798.38) * 2 * np.pi / 86400.0
    md = md0.anelastic(w_an, alpha=opts.alpha, Tm=opts.tm, T0=opts.t0, dispersion=opts.dispersion) if opts.anelastic else md0
    cen = opts.flattening and opts.centrifugal
    if opts.rotation:
        kw = dict(omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w, wahr_frame=(m != 2))
    else:
        kw = dict(omega_f=w)
    if opts.resolution is not None:
        layer_K, layer_N = resolution_per_layer(md, opts.resolution)
    else:
        layer_K = layer_N = None
    S = TidalSolver(md, geo, m=m, K=opts.K, N=opts.N, fluid_N=opts.fluidcore_N, fluid_K=opts.fluidcore_K,
                    layer_K=layer_K, layer_N=layer_N,
                    omega_rot=OMEGA_EARTH if cen else 0.0,
                    dyn_fluid=True, consistent_reference=cen, frame=opts.frame,
                    nonhydro=nh.state if nh is not None else None, **kw)
    r = S.solve()
    return S, r


def _surface_fields(S, th, deriv=False):
    """u_r, u_theta, u_lambda at s = a (or their d/ds), rigid rotation not removed."""
    from .basis import ybar, cheb_basis
    md = S.model
    iS = md.n_layers - 1
    Tp, T1, _ = cheb_basis(np.array([1.0]), S.Nl[iS])
    sc = 2.0 / (md.r1[iS] - md.r0[iS])
    v = {}
    for j, fld in enumerate(S.fields):
        c = S.layer_coef(iS, j)
        v[fld] = (c @ T1[:, 0] * sc) if deriv else (c @ Tp[:, 0])
    m = S.m
    ur = sum(v[('U', n)] * ybar(n, m, th, 0)[0] for n in S.deg)
    ut, up = 0 * th, 0 * th
    for n in S.degV:
        y1 = ybar(n, m, th, 1)[1]
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        ut, up = ut + v[('V', n)] * y1, up - v[('V', n)] * ys
    for n in S.degW:
        y1 = ybar(n, m, th, 1)[1]
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        ut, up = ut - v[('W', n)] * ys, up + v[('W', n)] * y1
    return ur, ut, up


def parameters(S, r, opts):
    """IERS-type parameters of a solution in the conventions of opts (complex numbers)."""
    from .iers import iers_fit_fields
    md, geo = S.model, S.geom
    a = md.a
    th = np.linspace(0.03, np.pi - 0.03, 600)
    if opts.flattening:
        from .geometry import surface_eps, equatorial_radius
        f, df = surface_eps(geo, md)
        a_e = equatorial_radius(geo, md)
    else:
        f, df, a_e = 0.0, 0.0, a
    raw = _surface_fields(S, th)
    body = list(S.surface_field(th))                       # rigid rotation removed (frame)
    if opts.surface == 'mean':                             # u(a) - delta / R_s d_s u (first order)
        rig = [x - y for x, y in zip(raw, body)]
        d = _surface_fields(S, th, deriv=True)
        db = [x - y / a for x, y in zip(d, rig)]           # d/dr of a rigid rotation = value / r
        P2 = 0.5 * (3 * np.cos(th)**2 - 1)
        delta = -2.0 / 3.0 * f * a * P2
        Rs = 1 - 2.0 / 3.0 * (f + a * df) * P2
        body = [x - delta / Rs * y for x, y in zip(body, db)]
    if opts.direction == 'normal':                         # angle between normal and radius
        dd = f * np.sin(2 * th)
        ur, ut, up = body
        body = [ur - dd * ut, ut - dd * ur, up]
    norm = a_e**4 / a**3 if opts.norm == 'equatorial' else a
    fit = iers_fit_fields(S.m, th, *(v / norm for v in body),
                          f_geodetic=f if opts.latitude == 'geodetic' else 0.0)
    out = {k: complex(fit[k]) for k in ('h0', 'h2', 'l0', 'l1', 'l2', 'hp', 'lp') if k in fit}
    out['k'] = complex(r['kk'])
    out['misfit'] = float(max(fit['misR'] / max(fit['scaleR'], 1e-300),
                              fit['misT'] / max(fit['scaleT'], 1e-300)))
    return out


def spherical_degree(opts, n):
    """h, l of degree n (> 2) for the spherical model at the M2 frequency."""
    from .solver import TidalSolver
    md0 = _model(opts)[0]
    w = F_M2 * 2 * np.pi / 86400.0
    md = md0.anelastic(w, alpha=opts.alpha, Tm=opts.tm, T0=opts.t0, dispersion=opts.dispersion) if opts.anelastic else md0
    S = TidalSolver(md, m=0, N=opts.N, n_force=n, degrees=([n], []), omega_f=w)
    r = S.solve()
    return {'h0': complex(r['h']), 'l0': complex(r['l']), 'k': complex(r['kk'])}


def _job(args):
    opts, n, m, f = args
    if n != 2:
        return spherical_degree(opts, n)
    if abs(f) < 1e-9:
        return {k: complex(v) for k, v in IERS_PERMANENT.items()} | {'note': 'IERS nominal (permanent tide)'}
    if opts.core_median:
        runs = [_job((replace(opts, fluidcore_N=int(N), core_median=None), n, m, f)) for N in opts.core_median]
        out = {}
        for k, v in runs[0].items():
            if isinstance(v, complex):
                vals = np.array([r[k] for r in runs])
                out[k] = complex(np.median(vals.real), np.median(vals.imag))
            else:
                out[k] = v
        dev = [abs(r[k] - out[k]) for r in runs for k, v in out.items() if isinstance(v, complex) and k != 'misfit']
        out['core_spread'] = float(max(dev)) if dev else 0.0
        return out
    if m == 1 and abs(f / F_TILT - 1) < TILT_WINDOW:
        # the tilt-over frequency (sigma = 1, K1): the free rigid tilt makes the system singular there,
        # and the solution is decided by round-off; the regular part is linear across sigma = 1
        # (the FCN is 2e-3 away), so take the mean of sigma = 1 -+ TILT_WINDOW (error ~1e-6)
        ps = [parameters(*solve_wave(opts, m, F_TILT * (1 + d)), opts) for d in (-TILT_WINDOW, TILT_WINDOW)]
        return {k: 0.5 * (ps[0][k] + ps[1][k]) for k in ps[0]}
    S, r = solve_wave(opts, m, f)
    return parameters(S, r, opts)


# ---------------------------------------------------------------------------- tables
def tamura_waves():
    """(degree, Doodson multipliers, frequency [cpd]) of the waves of etide's Tamura catalogue."""
    from .etide import Tamura
    cat = Tamura()
    return [(int(n), [int(x) for x in d], float(f)) for n, d, f in zip(cat.n, cat.dood, cat.f_cpd)]


def love_table(opts, waves, jobs=1, progress=True):
    """Love numbers for waves = [(n, doodson or None, f_cpd, m)] -> JSON-ready dict."""
    t0 = time.time()
    spherical = {}
    tasks, index = [], []
    for i, (n, dood, f, m) in enumerate(waves):
        if n != 2:
            if n not in spherical:
                spherical[n] = len(tasks)
                tasks.append((opts, n, 0, F_M2))
            index.append(spherical[n])
        else:
            index.append(len(tasks))
            tasks.append((opts, n, m, f))
    results = [None] * len(tasks)
    if jobs > 1:
        # one BLAS thread per worker: N workers x M threads would oversubscribe the cores
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, '1')
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for j, res in enumerate(ex.map(_job, tasks, chunksize=4)):
                results[j] = res
                if progress and j % 50 == 0:
                    print(f'{j + 1}/{len(tasks)} solved ({time.time() - t0:.0f} s)', flush=True)
    else:
        for j, t in enumerate(tasks):
            results[j] = _job(t)
            if progress and j % 50 == 0:
                print(f'{j + 1}/{len(tasks)} solved ({time.time() - t0:.0f} s)', flush=True)
    entries = []
    for (n, dood, f, m), j in zip(waves, index):
        e = dict(n=n, m=m, f_cpd=f)
        if dood is not None:
            e['doodson'] = dood
        res = dict(results[j])
        if n != 2:
            res['note'] = 'spherical, at the M2 frequency'
        for k, v in res.items():
            e[k] = [v.real, v.imag] if isinstance(v, complex) else v
        entries.append(e)
    return dict(format=FORMAT, format_version=FORMAT_VERSION, metadata=metadata(opts, time.time() - t0),
                entries=entries)


def metadata(opts, runtime):
    from . import __version__
    mp = model_path(opts.model)
    return dict(package='earth-tides', version=__version__,
                created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                model_file=mp.name, model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
                options={k: v for k, v in asdict(opts).items() if k != 'conventions'},
                nonhydrostatic=_model(opts)[2].summary() if opts.nonhydrostatic is not None else None,
                conventions=opts.conventions, iers_conventions=opts.is_iers(),
                units=dict(f_cpd='cycles per day',
                           parameters='complex [re, im]; h, l: equatorial radius; k: mean radius'),
                runtime_s=round(runtime, 1))


def default_name(opts, waves='tamura'):
    """File name built from the options (the metadata, not the name, is authoritative)."""
    parts = [Path(opts.model).stem, 'anel' if opts.anelastic else 'elas',
             'rot' if opts.rotation else 'norot', 'flat' if opts.flattening else 'sph']
    if opts.flattening and opts.figure != 'level':
        parts.append('first' if opts.figure == 'first-order' else opts.figure)
    if not opts.centrifugal:
        parts.append('nocentr')
    if opts.nonhydrostatic is not None:
        parts.append('nonhydro')
    if not opts.is_iers():
        parts += [opts.frame, opts.latitude[:4], opts.direction, opts.surface, opts.norm[:4]]
    return 'love_' + '_'.join(parts + [waves]) + '.json'


def write(table, path):
    with open(path, 'w') as fp:
        json.dump(table, fp, indent=1)


def read(path):
    """Read a Love-number table; complex parameters come back as complex numbers."""
    t = json.load(open(path))
    if t.get('format') != FORMAT:
        raise ValueError(f'{path}: not an {FORMAT} table')
    if t.get('format_version', 0) > FORMAT_VERSION:
        raise ValueError(f'{path}: format version {t["format_version"]} is newer than this package')
    for e in t['entries']:
        for k in KEYS:
            if k in e:
                e[k] = complex(*e[k])
    return t


# ---------------------------------------------------------------------------- wave groups
# ETERNA wave groups [cpd]: the example set of the ETERNA 3.4 manual (Wenzel 1997, ETERNA34.HLP,
# WAVEGROUPI), which puts P1, S1, K1, psi1 and phi1 in one K1 group, plus one long-period group.
ETERNA_GROUPS = [('long', 0.0, 0.501369), ('Q1', 0.501370, 0.911390), ('O1', 0.911391, 0.947991),
                 ('M1', 0.947992, 0.981854), ('K1', 0.981855, 1.023622),
                 ('J1', 1.023623, 1.057485), ('OO1', 1.057486, 1.470243),
                 ('2N2', 1.470244, 1.880264), ('N2', 1.880265, 1.914128),
                 ('M2', 1.914129, 1.950419), ('L2', 1.950420, 1.984282),
                 ('S2', 1.984283, 2.451943), ('M3M6', 2.451944, 7.000000)]
# The same with the K1 group split at the midpoints between P1, S1, K1, psi1 and phi1, so that the
# free core nutation resonance is followed (for displacements: 0.1 mm rms of the per-wave result at
# mid-latitudes, against 2 mm rms with the single K1 group).
ETERNA_SPLIT_GROUPS = (ETERNA_GROUPS[:4] +
                       [('P1', 0.981855, 0.998631), ('S1', 0.998632, 1.001369),
                        ('K1', 1.001370, 1.004107), ('PSI1', 1.004108, 1.006845),
                        ('PHI1', 1.006846, 1.023622)] + ETERNA_GROUPS[5:])
BUILTIN_GROUPS = {'eterna': ETERNA_GROUPS, 'eterna-split': ETERNA_SPLIT_GROUPS}
DEFAULT_GROUPS = DATA / 'groups_default.txt'   # tools/design_groups.py: 21 groups, < 0.03 mm


def read_groups(path):
    """Wave groups from a file: ETERNA lines 'WAVEGROUPI= f1 f2 gain phase name' or 'f1 f2 name'."""
    groups = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if not line:
            continue
        if line.upper().startswith('WAVEGROUPI='):
            w = line.split('=', 1)[1].split()
            groups.append((w[4] if len(w) > 4 else f'G{len(groups) + 1}', float(w[0]), float(w[1])))
        else:
            w = line.split()
            groups.append((w[2] if len(w) > 2 else f'G{len(groups) + 1}', float(w[0]), float(w[1])))
    return groups


def group_table(opts, groups, jobs=1, progress=True):
    """Love numbers per wave group, as in tidal-gravity practice (ETERNA/PREDICT): one solve per
    group at the frequency of its largest degree-2 wave, assigned to every degree-2 wave of the
    group (degrees 3 and 4 and the permanent tide as in love_table)."""
    from .etide import Tamura
    cat = Tamura()
    gname = []
    for n, f in zip(cat.n, cat.f_cpd):
        g = [name for name, f1, f2 in groups if f1 - 1e-9 <= abs(f) <= f2 + 1e-9]
        if n == 2 and abs(f) > 1e-9 and len(g) != 1:
            raise ValueError(f'wave at {f:.6f} cpd is in {len(g)} groups')
        gname.append(g[0] if g else None)
    reps = {}
    for i, (n, f) in enumerate(zip(cat.n, cat.f_cpd)):
        if n == 2 and abs(f) > 1e-9 and (gname[i] not in reps or abs(cat.A[i]) > abs(cat.A[reps[gname[i]]])):
            reps[gname[i]] = i
    order = sorted(reps, key=lambda g: cat.f_cpd[reps[g]])
    waves = [(2, [int(x) for x in cat.dood[reps[g]]], float(cat.f_cpd[reps[g]]), int(cat.m[reps[g]]))
             for g in order]
    t = love_table(opts, waves, jobs=jobs, progress=progress)
    rep = dict(zip(order, t['entries']))
    full = love_table(opts, [(int(n), [int(x) for x in d], float(f), int(d[0]))
                             for n, d, f in zip(cat.n, cat.dood, cat.f_cpd) if n != 2 or abs(f) < 1e-9],
                      progress=False)
    special = {(e['n'], tuple(e['doodson'])): e for e in full['entries']}
    entries = []
    for i, (n, d, f) in enumerate(zip(cat.n, cat.dood, cat.f_cpd)):
        key = (int(n), tuple(int(x) for x in d))
        if key in special:
            e = dict(special[key])
        else:
            e = {k: v for k, v in rep[gname[i]].items() if k not in ('n', 'm', 'f_cpd', 'doodson')}
            e.update(n=int(n), m=int(d[0]), f_cpd=float(f), doodson=list(key[1]))
            e['group'] = gname[i]
        entries.append(e)
    t['entries'] = entries
    t['metadata']['groups'] = [dict(name=g, f1=f1, f2=f2,
                                    representative_f_cpd=float(cat.f_cpd[reps[g]]) if g in reps else None)
                               for g, f1, f2 in groups]
    t['metadata']['runtime_s'] = round(t['metadata']['runtime_s'] + full['metadata']['runtime_s'], 1)
    return t


# ---------------------------------------------------------------------------- set-up file
TEMPLATE = """\
# earth-tides set-up file (TOML), read by: earthtides-love --setup FILE
# Every key is optional; the values below are the defaults unless marked "example".

model = "PREM"              # Earth model: file or PREM, PREMs, HOMOG
anelastic = true            # absorption-band anelasticity (Q from the model file)
alpha = 0.15                # frequency exponent of Q beyond tm
tm = 200.0                  # [s] end of the constant-Q band
dispersion = "cot"          # "cot" (exact for power-law Q, IERS) or "astyanax"
t0 = 1.0                    # reference period of the moduli [s]: 1 for PREM; 200 = the IERS assumption
core_median = [16, 20, 24, 30]   # median over these N in the liquid core (default); [] = one solve
rotation = true             # Coriolis force and inertia
flattening = true           # hydrostatic flattening of the level surfaces
figure = "level"            # "level" (level surfaces of the reference potential: second order in the
                            # flattening and beyond) or "first-order" (Clairaut; FCN of PREM 1.1 days longer)
centrifugal = true          # centrifugal force in the reference gravity
K = 4                       # spheroidal degrees per field
N = 24                      # Chebyshev polynomials per layer
# fluidcore_N = 40          # N in the fluid core only (it alone needs it; default: N)
# fluidcore_K = 12          # K in the fluid core only (default: K; keep K >= 6 for the diurnal band)
# K and N per layer (overrides the four lines above): rules by depth of the layer midpoint and by
# fluid/solid, later rules win (solver.resolution_per_layer); the same as JSON with --resolution
# [resolution]
# K = 6
# N = 24
# rules = [{depth_km = [0, 220], K = 4, N = 16}, {fluid = true, K = 12, N = 30}]

# conventions of the IERS parametrization (etide needs these values)
latitude = "geocentric"     # or "geodetic"
direction = "radial"        # or "normal"
surface = "deformed"        # or "mean"
norm = "equatorial"         # or "mean"
frame = "mantle"            # or "surface"

# Non-hydrostatic Earth (example; leave the table out for a hydrostatic Earth). The excess
# flattening is produced by zonal density loads in mantle shells, supported elastically on the
# flattened rotating background (manual, Sect. 'A non-hydrostatic Earth'). Targets: J2 (excess J2),
# surface and cmb (excess equatorial minus polar radius, m), e_f (excess dynamical ellipticity of
# the fluid core). An exact fit needs as many targets as shells; with sigma and prior the fit is
# regularised (targets weighted by sigma, loads kept small by prior, relative density).
# [nonhydrostatic]
# support = "elastic"       # or "viscous": instantaneous mantle flow, mu = viscosity_scale x eta,
#                           # viscosity_scale = 2e6 (Pa), eta_lower = 50, eta_upper = 1 (relative),
#                           # 80 km elastic lid; needs far larger loads for the same J2
# shells_km = [[3480, 5701], [5701, 6346.6]]     # lower mantle, upper mantle (radii)
# targets = {J2 = 1.14e-5, e_f = 1.238e-4}       # Chambat et al. 2010; FCN 431.4 sd (anelastic PREM,
#                                                # level figure; the e_f of MHB 2002, 9.73e-5, gives 437 sd)
# sigma = {J2 = 1e-7, e_f = 5e-6}
# prior = 5e-3
# model_static = "PREMs"    # the model with a weak-solid core for the static support (PREM: PREMs)

# What to compute (command-line options override these)
[run]
groups = "default"          # one Love number per wave group: default | eterna | eterna-split | FILE
# all_waves = true          # instead: every wave of the Tamura catalogue
# freq = [0.92700, 1.00274] # instead: these frequencies [cpd] ...
# order = 1                 # ... of this order m
# output = "love.json"
"""

RUN_KEYS = ('groups', 'all_waves', 'freq', 'order', 'output', 'jobs')


def read_setup(path):
    """Options and the [run] table from a TOML set-up file (TEMPLATE)."""
    try:
        import tomllib                 # Python 3.11+
    except ImportError:
        import tomli as tomllib        # Python 3.10
    with open(path, 'rb') as fp:
        d = tomllib.load(fp)
    run = d.pop('run', {})
    bad = [k for k in run if k not in RUN_KEYS]
    if bad:
        raise ValueError(f'unknown [run] keys {bad}; allowed {RUN_KEYS}')
    allowed = {f for f in Options.__dataclass_fields__ if f != 'conventions'}
    bad = [k for k in d if k not in allowed]
    if bad:
        raise ValueError(f'unknown set-up keys {bad}; allowed {sorted(allowed)}')
    if 'resolution' not in d and 'fluidcore_N' not in d:
        d.setdefault('core_median', list(CORE_MEDIAN))
    if d.get('core_median') in ([], False):
        d['core_median'] = None
    return Options(**d), run


CORE_MEDIAN = (16, 20, 24, 30)     # default of earthtides-love and of its set-up files: median over these core N

# ---------------------------------------------------------------------------- command line
def main(argv=None):
    p = argparse.ArgumentParser(prog='earthtides-love', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='PREM', help='Earth model: file or PREM, PREMs, HOMOG')
    p.add_argument('--all-waves', action='store_true',
                   help='a Love number for every wave of the Tamura (1987) catalogue (837 solves) '
                        'instead of one per wave group')
    p.add_argument('--waves', choices=['tamura'], help=argparse.SUPPRESS)   # = --all-waves
    p.add_argument('--groups', metavar='default|eterna|eterna-split|FILE', default='default',
                   help='one Love number per wave group (default): the 21 groups of earth-tides plus HIGH (degrees 3, 4) '
                        '(error < 0.03 mm against all waves), the ETERNA example set, the same with '
                        'the K1 group split, or a file of WAVEGROUPI lines; the table always covers '
                        'every Tamura wave')
    p.add_argument('--freq', type=float, nargs='+', help='frequencies [cycles per day] instead of the Tamura catalogue')
    p.add_argument('--order', type=int, choices=[0, 1, 2], help='order m for --freq')
    p.add_argument('--elastic', action='store_true', help='no anelasticity')
    p.add_argument('--dispersion', default='cot', choices=['cot', 'astyanax'],
                   help='anelastic dispersion beyond Tm: cot(alpha pi/2) (exact for power-law Q, '
                        'IERS; default) or 2/(pi alpha) (astyanax)')
    p.add_argument('--t0', type=float, default=1.0,
                   help='reference period [s] of the moduli of the Earth model: constant Q from t0 to 200 s, '
                        'power law beyond (default 1 s, PREM; 200 reproduces the dispersion assumed for the '
                        'anelastic values of the IERS Conventions)')
    p.add_argument('--no-rotation', action='store_true', help='no Coriolis force (inertia only)')
    p.add_argument('--sphere', action='store_true', help='no flattening')
    p.add_argument('--figure', default='level', choices=['first-order', 'level'],
                   help='figure of the flattened Earth: first order in the flattening (Clairaut) or level '
                        'surfaces of the reference potential (second order and beyond)')
    p.add_argument('--no-centrifugal', action='store_true', help='no centrifugal force in the reference gravity')
    p.add_argument('--frame', default='mantle', choices=['mantle', 'surface'])
    p.add_argument('--latitude', default='geocentric', choices=['geocentric', 'geodetic'])
    p.add_argument('--direction', default='radial', choices=['radial', 'normal'])
    p.add_argument('--surface', default='deformed', choices=['deformed', 'mean'])
    p.add_argument('--norm', default='equatorial', choices=['equatorial', 'mean'])
    p.add_argument('-K', type=int, default=4, help='spheroidal degrees per field')
    p.add_argument('-N', type=int, default=24, help='Chebyshev polynomials per layer')
    p.add_argument('--fluidcore_N', type=int, default=None,
                   help='Chebyshev polynomials per layer in the fluid core only (default: N)')
    p.add_argument('--fluidcore_K', type=int, default=None,
                   help='degrees per field in the fluid core only (default: K)')
    p.add_argument('--core-median', type=int, nargs='+', default=list(CORE_MEDIAN), metavar='N',
                   help='solve each wave with these N in the liquid core and take the median of every '
                        'parameter (default 16 20 24 30): rejects the pseudo-modes of an inviscid core, which '
                        'fall on different waves at each resolution (manual, Sect. "The viscous core")')
    p.add_argument('--single-core', action='store_true',
                   help='one solve per wave with --fluidcore_N (or N) in the core, no median')
    p.add_argument('--resolution', metavar='FILE', help='JSON file with K and N per layer (rules by '
                   'depth and fluid/solid; see solver.resolution_per_layer); overrides -K -N '
                   '--fluidcore_K --fluidcore_N')
    p.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    p.add_argument('-o', '--output', help='output file (default: built from the options)')
    p.add_argument('--setup', metavar='FILE', help='TOML set-up file with all physics options, the '
                   'non-hydrostatic Earth and a [run] table (see --template); the physics flags '
                   'above are then not allowed')
    p.add_argument('--template', action='store_true', help='print an annotated set-up file and exit')
    a = p.parse_args(argv)
    if a.template:
        print(TEMPLATE, end='')
        return
    if a.setup:
        physics = ('model', 'elastic', 'dispersion', 't0', 'core_median', 'single_core', 'no_rotation', 'sphere', 'no_centrifugal', 'frame',
                   'latitude', 'direction', 'surface', 'norm', 'K', 'N', 'fluidcore_N', 'fluidcore_K', 'resolution')
        given = [k for k in physics if getattr(a, k) != p.get_default(k)]
        if given:
            p.error(f'with --setup, set {given} in the set-up file')
        opts, run = read_setup(a.setup)
        for k in ('groups', 'output', 'jobs'):
            if k in run and getattr(a, k) == p.get_default(k):
                setattr(a, k, run[k])
        if run.get('all_waves') and not a.freq:
            a.all_waves = True
        if 'freq' in run and not (a.freq or a.all_waves):
            a.freq, a.order = run['freq'], run.get('order', a.order)
    else:
        opts = Options(model=a.model, anelastic=not a.elastic, rotation=not a.no_rotation,
                       flattening=not a.sphere, figure=a.figure, centrifugal=not a.no_centrifugal, frame=a.frame,
                       latitude=a.latitude, direction=a.direction, surface=a.surface, norm=a.norm,
                       K=a.K, N=a.N, fluidcore_N=a.fluidcore_N,
                       fluidcore_K=a.fluidcore_K,
                       resolution=json.load(open(a.resolution)) if a.resolution else None, dispersion=a.dispersion,
                       t0=a.t0,
                       core_median=None if (a.single_core or a.resolution or a.fluidcore_N) else a.core_median)
    if a.waves == 'tamura':
        a.all_waves = True
    if not (a.all_waves or a.freq):
        groups = (read_groups(DEFAULT_GROUPS) if a.groups == 'default'
                  else BUILTIN_GROUPS.get(a.groups) or read_groups(a.groups))
        out = a.output or default_name(opts, 'groups_' + (a.groups if a.groups in ('default', *BUILTIN_GROUPS)
                                                          else Path(a.groups).stem))
        table = group_table(opts, groups, jobs=a.jobs)
        write(table, out)
        nsolve = sum(1 for g in table['metadata']['groups'] if g['representative_f_cpd'] is not None)
        nhigh = len(groups) - nsolve
        print(f'{len(groups)} groups: {nsolve} solved, {nhigh} with only degrees 3 and 4 (no solve)')
        print(f'{len(table["entries"])} entries -> {out} ({table["metadata"]["runtime_s"]:.0f} s)')
        return
    if a.all_waves:
        waves = [(n, d, f, d[0]) for n, d, f in tamura_waves()]
        tag = 'allwaves'
    elif a.freq:
        if a.order is None:
            p.error('--freq needs --order')
        waves = [(2, None, f, a.order) for f in a.freq]
        tag = f'm{a.order}_{len(a.freq)}freq'
    else:
        p.error('internal: no mode')
    out = a.output or default_name(opts, tag)
    table = love_table(opts, waves, jobs=a.jobs)
    write(table, out)
    print(f'{len(table["entries"])} entries -> {out} ({table["metadata"]["runtime_s"]:.0f} s)')


if __name__ == '__main__':
    main()
