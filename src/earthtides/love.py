# -*- coding: utf-8 -*-
#
# Love numbers of a flattened, rotating, anelastic Earth for a list of tidal
# waves, written to a JSON table (the input of etide).
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

Command line: earthtides-love [--groups default|eterna|eterna-split|FILE | --all-waves] -o love.json
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
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
IERS_PERMANENT = dict(h0=0.6078, h2=-0.0006, l0=0.0847, l2=0.0002)


@dataclass
class Options:
    """Physics, numerics and conventions of a Love-number table (stored in its metadata)."""
    model: str = 'PREM'
    anelastic: bool = True
    alpha: float = 0.15
    tm: float = 200.0
    dispersion: str = 'cot'
    rotation: bool = True
    flattening: bool = True
    centrifugal: bool = True
    frame: str = 'mantle'
    latitude: str = 'geocentric'
    direction: str = 'radial'
    surface: str = 'deformed'
    norm: str = 'equatorial'
    K: int = 4
    N: int = 24
    f_dyn: float = 0.005
    conventions: dict = field(init=False)

    def __post_init__(self):
        for key, allowed in (('dispersion', ('cot', 'astyanax')),
                             ('frame', ('mantle', 'surface')), ('latitude', ('geocentric', 'geodetic')),
                             ('direction', ('radial', 'normal')), ('surface', ('deformed', 'mean')),
                             ('norm', ('equatorial', 'mean'))):
            if getattr(self, key) not in allowed:
                raise ValueError(f'{key} must be one of {allowed}')
        self.conventions = {k: getattr(self, k) for k in IERS_CONVENTIONS}

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
    from .model import EarthModel
    from .geometry import ClairautGeometry
    key = (opts.model, opts.flattening)
    if key not in _CACHE:
        md = EarthModel(str(model_path(opts.model)))
        _CACHE[key] = (md, ClairautGeometry(md) if opts.flattening else None)
    return _CACHE[key]


def solve_wave(opts, m, f_cpd):
    """Solve the degree-2 tide of order m and frequency f_cpd [cycles per day]."""
    from .geometry import OMEGA_EARTH
    from .solver import TidalSolver
    md0, geo = _model(opts)
    w = max(f_cpd, opts.f_dyn) * 2 * np.pi / 86400.0
    w_an = max(f_cpd, 1 / 6798.38) * 2 * np.pi / 86400.0
    md = md0.anelastic(w_an, alpha=opts.alpha, Tm=opts.tm, dispersion=opts.dispersion) if opts.anelastic else md0
    cen = opts.flattening and opts.centrifugal
    if opts.rotation:
        kw = dict(omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w, wahr_frame=(m != 2))
    else:
        kw = dict(omega_f=w)
    S = TidalSolver(md, geo, m=m, K=opts.K, N=opts.N, omega_rot=OMEGA_EARTH if cen else 0.0,
                    dyn_fluid=True, consistent_reference=cen, frame=opts.frame, **kw)
    r = S.solve()
    return S, r


def _surface_fields(S, th, deriv=False):
    """u_r, u_theta, u_lambda at s = a (or their d/ds), rigid rotation not removed."""
    from .basis import ybar, cheb_basis
    md = S.model
    iS = md.n_layers - 1
    N, nb = S.N, S.nfld * S.N
    Tp, T1, _ = cheb_basis(np.array([1.0]), N)
    sc = 2.0 / (md.r1[iS] - md.r0[iS])
    v = {}
    for j, fld in enumerate(S.fields):
        c = S.coef[iS * nb + j * N: iS * nb + (j + 1) * N]
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
        e, de, _ = geo.eps(md.n_layers - 1, np.array([a]))
        f, df = float(e[0]), float(de[0])
    else:
        f, df = 0.0, 0.0
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
    norm = (a * (1 + f / 3.0))**4 / a**3 if opts.norm == 'equatorial' else a
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
    md0, _ = _model(opts)
    w = F_M2 * 2 * np.pi / 86400.0
    md = md0.anelastic(w, alpha=opts.alpha, Tm=opts.tm, dispersion=opts.dispersion) if opts.anelastic else md0
    S = TidalSolver(md, m=0, N=opts.N, n_force=n, degrees=([n], []), omega_f=w)
    r = S.solve()
    return {'h0': complex(r['h']), 'l0': complex(r['l']), 'k': complex(r['kk'])}


def _job(args):
    opts, n, m, f = args
    if n != 2:
        return spherical_degree(opts, n)
    if abs(f) < 1e-9:
        return {k: complex(v) for k, v in IERS_PERMANENT.items()} | {'note': 'IERS nominal (permanent tide)'}
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
                conventions=opts.conventions, iers_conventions=opts.is_iers(),
                units=dict(f_cpd='cycles per day',
                           parameters='complex [re, im]; h, l: equatorial radius; k: mean radius'),
                runtime_s=round(runtime, 1))


def default_name(opts, waves='tamura'):
    """File name built from the options (the metadata, not the name, is authoritative)."""
    parts = [Path(opts.model).stem, 'anel' if opts.anelastic else 'elas',
             'rot' if opts.rotation else 'norot', 'flat' if opts.flattening else 'sph']
    if not opts.centrifugal:
        parts.append('nocentr')
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
    p.add_argument('--no-rotation', action='store_true', help='no Coriolis force (inertia only)')
    p.add_argument('--sphere', action='store_true', help='no flattening')
    p.add_argument('--no-centrifugal', action='store_true', help='no centrifugal force in the reference gravity')
    p.add_argument('--frame', default='mantle', choices=['mantle', 'surface'])
    p.add_argument('--latitude', default='geocentric', choices=['geocentric', 'geodetic'])
    p.add_argument('--direction', default='radial', choices=['radial', 'normal'])
    p.add_argument('--surface', default='deformed', choices=['deformed', 'mean'])
    p.add_argument('--norm', default='equatorial', choices=['equatorial', 'mean'])
    p.add_argument('-K', type=int, default=4, help='spheroidal degrees per field')
    p.add_argument('-N', type=int, default=24, help='Chebyshev polynomials per layer')
    p.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    p.add_argument('-o', '--output', help='output file (default: built from the options)')
    a = p.parse_args(argv)
    opts = Options(model=a.model, anelastic=not a.elastic, rotation=not a.no_rotation,
                   flattening=not a.sphere, centrifugal=not a.no_centrifugal, frame=a.frame,
                   latitude=a.latitude, direction=a.direction, surface=a.surface, norm=a.norm,
                   K=a.K, N=a.N, dispersion=a.dispersion)
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
