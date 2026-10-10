#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Reproduce the numbers, tables and figures of the earth-tides paper.
#
# This file is part of earth-tides 0.1.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 10/10/2026 Machiel Bos
#===============================================================================
"""Reproduce the results of

    Bos, M. S., "What the IERS body-tide model assumes: Love numbers of a flattened, rotating,
    anelastic Earth", Journal of Geodesy (submitted), and its Online Resource 1.

Every job runs one script of scripts/ (or the figure script, or a worked example) in a temporary
copy of this tree, so the stored results in results/ are never overwritten: they are the reference.
The outputs of a job are then compared with the stored ones number by number: a printed number may
differ by one unit in its last digit (or 1e-8 relative); the text must be equal. README.md lists
which job produces which table, figure and number of the paper.

Levels
  --quick     (default) the fast jobs and all figures redrawn from the stored data, ~3 minutes
  --full      every job that needs only this package (Python, NumPy, SciPy, Matplotlib), ~1.5 h
  --external  in addition the jobs that need data or programs not distributed here; each is
              skipped, with the reason, if its inputs are not found:
                ocean tide grids (FES2014b, FES2022b, GOT5.5, TPXO10-atlas) in --grids DIR
                the IERS routine DEHANTTIDEINEL, compiled in variants, and the DE421 ephemeris
                  in --iers DIR (see tools/iers/), with skyfield installed
                the FES2004 coefficients of the IERS Conventions (fes2004.dat) in --iers DIR
              The ocean tide loading at the gravimeter stations (results/tidal_gravity/CARGA_*.csv)
              was computed with CARGA, which is not distributed; those files are inputs here.
Other options
  --list      list the jobs and exit
  --only NAME run only the named jobs (and nothing else)
  --keep DIR  keep the outputs in DIR (default: a temporary directory that is removed)
  --installed use the installed package (pip install earth-tides) instead of src/
  --update    copy the new outputs over the stored results (to refresh them after a code change)
Usage: python paper_scripts/run_all.py [--quick|--full|--external] [--only NAME ...]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# name, level, command (relative to the tree), outputs, requirement, paper items
#   level: quick | full | external;  requirement: None | 'grids' | 'iers' | 'fes2004'
#   an output 'stdout:NAME' is the printed output, stored as results/stdout/NAME.txt
JOBS = [
    # --- method and verification (Sects 3-4, Table 1, Online Resource S3)
    ('check_hard', 'quick', ['scripts/check_hard.py'], ['stdout:check_hard'], None,
     'Table 1: PREM as a sphere vs Bos & Scherneck (2013)'),
    ('check_m', 'quick', ['scripts/check_m.py'], ['stdout:check_m'], None,
     'Table 1: independence of m'),
    ('check_bump', 'quick', ['scripts/check_bump.py'], ['stdout:check_bump'], None,
     'Table 1, S3: distorted mesh'),
    ('exact_spheroid', 'quick', ['scripts/check_exact_spheroid.py'], ['results/exact_spheroid.txt'],
     'sympy', 'Table 1, S3: homogeneous elastic spheroid, cost f^2/2'),
    ('ex3', 'quick', ['src/earthtides/examples/ex3/homogeneous.py'],
     ['src/earthtides/examples/ex3/homogeneous.out'], None, 'Table 1: homogeneous spheres'),
    ('ex4', 'full', ['src/earthtides/examples/ex4/greff2005.py'],
     ['src/earthtides/examples/ex4/greff2005.out'], None, 'Table 1, S3: Greff-Lefftz et al. (2005)'),
    ('ex5', 'quick', ['src/earthtides/examples/ex5/hough1895.py'],
     ['src/earthtides/examples/ex5/hough1895.out'], None, "Table 1, Sect. 6.1: Hough's fluid core"),
    ('angular_momentum', 'quick', ['scripts/check_angular_momentum.py'], ['stdout:check_angular_momentum'],
     None, 'Table 1, S3: angular-momentum balance'),
    ('fcn_period', 'quick', ['scripts/fcn_period.py', 'anelastic'], ['stdout:fcn_period_anelastic'],
     None, 'Sect. 6.1, S4: FCN of anelastic hydrostatic PREM (460.0 sidereal days)'),
    ('convention_table', 'quick', ['scripts/convention_table.py'], ['results/convention_table.txt'],
     None, 'Sect. 8.3, Table S3: effect of the conventions'),
    ('check_chao1996', 'quick', ['scripts/check_chao1996.py'], ['stdout:check_chao1996'], None,
     'Table 5, Sect. 8.2: admittance from the angular momenta of Chao et al. (1996)'),
    ('check_iers_tables', 'quick', ['scripts/check_iers_tables.py'], ['stdout:check_iers_tables'], None,
     'Sect. 8.2: IERS Tables 6.5a, 7.2, 7.3a reproduced only with the empirical admittance'),
    ('ducarme', 'quick', ['scripts/check_tidal_gravity_ducarme.py'],
     ['results/tidal_gravity/check_ducarme.txt'], None, 'Sect. 6.4: K1/O1 ratios of Ducarme et al.'),
    ('tidal_gravity_k1', 'quick', ['scripts/check_tidal_gravity_k1.py'],
     ['stdout:check_tidal_gravity_k1'], None, 'Sect. 6.4: K1/O1 with our ocean loading (CARGA inputs)'),
    ('figures', 'quick', ['paper/make_figures.py', '--replot'], ['figures'], None,
     'Figs 1-5, S1-S3 (from the stored data)'),
    # --- Love numbers and the free core nutation (Sects 5-6, Tables 2-4)
    ('diurnal', 'full', ['scripts/run_diurnal.py'], ['results/diurnal_PREM.txt', 'results/figdata_fcn.npz'],
     None, 'Table 2, Fig. 4: diurnal band, resonance sweep'),
    ('compare_models', 'full', ['scripts/compare_models.py'], ['results/compare_models.txt'], None,
     'Sect. 5.3: PREM vs STW105'),
    ('anelastic', 'full', ['scripts/run_anelastic.py'], ['results/anelastic_PREMs.txt'], None,
     'Sect. 5.3: anelastic change of the latitude terms'),
    ('coriolis', 'full', ['scripts/run_coriolis.py'], ['results/coriolis_PREMs.txt'], None,
     'Sect. 5.2: Coriolis part of l(1) and l\''),
    ('scan_delta_plus', 'full', ['scripts/scan_delta_plus.py'], ['stdout:scan_delta_plus'], None,
     'Sect. 6.4: gravimetric factors vs Dehant et al. (1999)'),
    ('gravimetric_spinup', 'full', ['scripts/gravimetric_factor_spinup.py'],
     ['results/gravimetric_factor_spinup.txt'], None, 'Sect. 6.4: gravimetric factors, K1/O1 0.98300'),
    ('gravimetric_fcn430', 'full', ['scripts/gravimetric_factor_fcn430.py'],
     ['results/gravimetric_factor_fcn430.txt'], None, 'Sect. 6.4: resonance residues, rho_pred'),
    ('k1o1_sensitivity', 'full', ['scripts/k1o1_sensitivity.py'], ['results/k1o1_sensitivity.txt'],
     None, 'Sect. 6.4: K1/O1 vs mantle rigidity'),
    ('love_table', 'full', ['-m', 'earthtides.love', '--jobs', '4', '-o', 'results/love_PREM_check.json'],
     ['results/love_PREM_check.json'], None, 'Sect. 7: default Love table of PREM (data/love_PREM.json)'),
    ('displacement_budget', 'full', ['scripts/displacement_budget.py'], ['results/displacement_budget.txt'],
     None, 'Online Resource S7, Table S2, Fig. S3: effect of each choice at four stations'),
    ('prem_iers_harmonics', 'full', ['scripts/prem_iers_harmonics.py'], ['results/prem_iers_harmonics.txt'],
     None, 'Sect. 7.1: PREM - IERS, rms and line amplitudes'),
    ('paper_numbers', 'full', ['scripts/paper_numbers.py'], ['results/paper_numbers.txt'], None,
     'numbers in the text not printed elsewhere (FCN periods, Table 4 residues, mesh, mm conversions)'),
    ('figure_data', 'full', ['paper/make_figures.py'], ['results/figdata.npz', 'results/figdata_toy.npz'],
     None, 'Figs 1-5, S1-S3 with their data recomputed'),
    # --- jobs that need external data or programs
    ('otl_fcn_resonance', 'external', ['scripts/otl_fcn_resonance.py'], ['results/otl_fcn_resonance.txt'],
     'grids', 'Sects 6.2-6.3, Tables 5-6: load resonance, admittance A21, resonant correction (FES2022b)'),
    ('station_error_budget', 'external', ['scripts/station_error_budget.py'],
     ['results/station_error_budget.txt', 'results/station_error_budget.npz'], 'grids',
     'Sect. 7.2, Fig. 5, Table 7: uncertainty at nine stations (four ocean tide models)'),
    ('iers_admittance', 'external', ['scripts/check_iers_admittance.py', '{IERS}/fes2004.dat'],
     ['stdout:check_iers_admittance'], 'fes2004', 'Table 5: FES2004 admittance in the IERS conventions'),
    ('station_admittance', 'external', ['scripts/station_admittance.py'], ['results/station_admittance.npz'],
     'iers', 'input of body_tide_check: admittance per HARDISP line (25 minutes)'),
    ('body_tide_check', 'external', ['scripts/body_tide_check.py', '{IERS}', 'all'],
     ['results/body_tide_check.txt'], 'iers', 'Table S1: IERS degree 2 and 3, step 2'),
    ('etide_vs_iers', 'external', ['scripts/etide_vs_iers.py', '{IERS}'], ['results/etide_vs_iers.txt'],
     'iers', 'Sect. 7, S6, Table S1: etide vs DEHANTTIDEINEL term by term'),
]

NUM = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
CPLX = re.compile(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)([+-][\d.]+(?:[eE][+-]?\d+)?)[ij]$')
TIMING = re.compile(r'\(\s*[\d.]+\s*s\)|\[[^\]]*:\s*\d+\s*s\]|\b\d+\.\d+\s*s\b|done\s+[\d.]+'
                    r'|\d+/\d+ solved')
TOL = 1e-8


def tokens(text):
    t = TIMING.sub('', text).replace(',', ' ').replace('=', ' = ').replace('/', ' / ')
    return t.translate(str.maketrans({c: ' ' for c in '\'"{}()[]'})).split()


def ulp(tok):
    m = NUM.match(tok.lstrip('+-'))
    dec = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
    exp = int(m.group(2)[1:]) if m.group(2) else 0
    return 10.0**(exp - dec)


def numbers(tok):
    t = tok.rstrip('%')
    m = CPLX.match(t)
    if m:
        return [m.group(1), m.group(2)]
    return [t] if NUM.match(t) else None


# Numbers that are round-off on every platform (|x| and |y| below ROUNDOFF) count as equal.
ROUNDOFF = 1e-12
# Lines left out of the comparison, per output file, with the reason.
IGNORE = {'hough1895.out': (re.compile(r'numerical mode'),
                            'spurious eigenvalues of the discretization; how many are found depends '
                            'on round-off (see the README of example 5)')}


# Files whose numbers vary between platforms by more than the printed precision: a function
# (reference value) -> allowed absolute difference, with the reason.
def _hough_tol(y):
    """Example 5: the FCN of Hough's Earth is the root of a nearly singular system and varies by
    ~1e-7 in frequency between BLAS builds; periods (1 / (f - 1), > 10) then vary by ~1e-4
    relative and the relative differences (< 0.5) by ~1e-4."""
    y = abs(y)
    return 1e-3 * y if y >= 10 else (3e-7 if y >= 0.5 else 2e-4)


FILE_TOL = {'hough1895.out': _hough_tol}
_FT = None


def same_number(x, y):
    fx, fy = float(x), float(y)
    if abs(fx) < ROUNDOFF and abs(fy) < ROUNDOFF:
        return True
    if _FT and abs(fx - fy) <= _FT(fy):
        return True
    if abs(fx - fy) <= 1.01 * max(ulp(x.lstrip('+-')), ulp(y.lstrip('+-'))):
        return True
    return abs(fx - fy) <= TOL * max(abs(fy), 1.0)


def compare_text(a, b, name=''):
    global _FT
    _FT = FILE_TOL.get(name)
    if name in IGNORE:
        a, b = ('\n'.join(l for l in t.splitlines() if not IGNORE[name][0].search(l)) for t in (a, b))
    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb):
        return [f'different length ({len(ta)} vs {len(tb)} tokens)']
    out = []
    for x, y in zip(ta, tb):
        if x == y:
            continue
        nx, ny = numbers(x), numbers(y)
        if nx and ny and len(nx) == len(ny) and all(same_number(u, v) for u, v in zip(nx, ny)):
            continue
        out.append(f'{y} -> {x}')
    return out


# Arrays of the figure data compared with an absolute tolerance (convergence errors at round-off)
# or only away from the equator (l_N divides by dY/dphi, which vanishes there; the figure masks it).
NPZ_ATOL = {'f2_err': 1e-10, 'f3_err': 1e-10}


def compare_file(new, ref):
    if ref.suffix == '.npz':
        import numpy as np
        a, b = np.load(new), np.load(ref)
        bad = []
        for k in b.files:
            if k not in a.files or a[k].shape != b[k].shape:
                bad.append(k)
                continue
            x, y = a[k], b[k]
            if y.dtype.kind not in 'fciu':
                if not np.array_equal(x, y):
                    bad.append(k)
                continue
            if k.startswith('f4_lN') and 'f4_th' in b.files:
                keep = np.abs(90 - np.degrees(b['f4_th'])) >= 4
                x, y = x[keep], y[keep]
            atol = next((v for p_, v in NPZ_ATOL.items() if k.startswith(p_)), 1e-12)
            if not np.allclose(x, y, rtol=1e-6, atol=atol):
                bad.append(k)
        return [f'array {k} differs' for k in bad]
    if ref.suffix == '.json':
        return compare_text(new.read_text().replace('[', ' ').replace(']', ' '),
                            ref.read_text().replace('[', ' ').replace(']', ' '))
    return compare_text(new.read_text(), ref.read_text(), ref.name)


def wpath(o, installed):
    """Path of a job file in the work tree (the examples move to examples/ for --installed)."""
    return o.replace('src/earthtides/examples/', 'examples/', 1) if installed else o


def copy_tree(dest, installed=False):
    keep = ('scripts', 'data', 'results', 'paper', 'tools', 'pyproject.toml') + (() if installed else ('src',))
    if installed:                                   # the examples only, not the package
        ex = ROOT / 'examples' if (ROOT / 'examples').is_dir() else ROOT / 'src' / 'earthtides' / 'examples'
        shutil.copytree(ex, dest / 'examples', ignore=shutil.ignore_patterns('__pycache__'))
    (dest / 'docs' / 'manual').mkdir(parents=True, exist_ok=True)   # figure helpers of the manual
    if (ROOT / 'docs' / 'manual' / 'make_figures.py').exists():
        shutil.copy(ROOT / 'docs' / 'manual' / 'make_figures.py', dest / 'docs' / 'manual')
    for k in keep:
        s = ROOT / k
        if s.is_dir():
            shutil.copytree(s, dest / k, ignore=shutil.ignore_patterns('__pycache__', '*.pdf',
                                                                       '*.aux', '*.log', '*.bbl'))
        elif s.exists():
            shutil.copy(s, dest / k)
    (dest / 'paper' / 'figures').mkdir(parents=True, exist_ok=True)


def requirement_ok(req, a):
    if req is None:
        return True, ''
    if req == 'sympy':
        try:
            import sympy  # noqa: F401
            return True, ''
        except ImportError:
            return False, 'sympy not installed'
    if req == 'grids':
        g = Path(a.grids)
        need = [g / m / 'amp.k1' for m in ('FES2014b', 'FES2022b', 'GOT5.5', 'TPXO10_atlas_v2')]
        miss = [str(p.parent) for p in need if not p.exists()]
        return not miss, 'ocean tide grids missing: ' + ', '.join(miss)
    if req == 'fes2004':
        ok = a.iers and (Path(a.iers) / 'fes2004.dat').exists()
        return bool(ok), 'fes2004.dat not found (--iers DIR)'
    if req == 'iers':
        ok = a.iers and (Path(a.iers) / 'de421.bsp').exists() and (Path(a.iers) / 'dehant').exists()
        try:
            import skyfield  # noqa: F401
        except ImportError:
            return False, 'skyfield not installed'
        return bool(ok), 'DE421 and the DEHANTTIDEINEL variants not found (--iers DIR)'
    return False, req


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--quick', action='store_const', dest='level', const='quick')
    g.add_argument('--full', action='store_const', dest='level', const='full')
    g.add_argument('--external', action='store_const', dest='level', const='external')
    p.add_argument('--list', action='store_true')
    p.add_argument('--only', nargs='+')
    p.add_argument('--keep')
    p.add_argument('--update', action='store_true',
                   help='write the new outputs over the stored results (after a code change)')
    p.add_argument('--installed', action='store_true',
                   help='use the installed earthtides package (pip install earth-tides), not src/')
    p.add_argument('--grids', default=str(Path.home() / 'projects/otl_map/surfaceloads'))
    p.add_argument('--iers', default=str(Path.home() / 'scratch/earthtides_iers'))
    a = p.parse_args()
    level = a.level or 'quick'
    order = {'quick': 0, 'full': 1, 'external': 2}
    if a.list:
        for name, lev, cmd, outs, req, what in JOBS:
            print(f'{name:22s} {lev:9s} {what}')
        return
    jobs = [j for j in JOBS if (a.only and j[0] in a.only) or (not a.only and order[j[1]] <= order[level])]
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix='earthtides_repro_'))
    if work.exists() and any(work.iterdir()):
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    copy_tree(work, a.installed)
    env = dict(os.environ, PYTHONPATH='' if a.installed else str(work / 'src'), PYTHONWARNINGS='ignore',
               MPLBACKEND='Agg')
    r = subprocess.run([PY, '-c', 'import earthtides; print(earthtides.__version__, earthtides.__file__)'],
                       cwd=work, env=env, capture_output=True, text=True)
    print(f'earthtides {r.stdout.strip() or r.stderr.strip().splitlines()[-1]}', flush=True)
    (work / 'results' / 'stdout').mkdir(exist_ok=True)
    report, t_all = [], time.time()
    try:
        for name, lev, cmd, outs, req, what in jobs:
            ok, why = requirement_ok(req, a)
            if not ok:
                report.append((name, 'skip', why))
                print(f'[skip] {name}: {why}', flush=True)
                continue
            for o in outs:                                # the stored output must not be read back
                if not o.startswith(('stdout:', 'figures')) and (work / wpath(o, a.installed)).exists():
                    (work / wpath(o, a.installed)).unlink()
            args = [PY] + [c.replace('{IERS}', str(a.iers)) for c in cmd]
            cwd = work
            if cmd[0].startswith('src/earthtides/examples/'):
                cwd = work / Path(wpath(cmd[0], a.installed)).parent
                args = [PY, Path(cmd[0]).name]
            t = time.time()
            r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
            dt = time.time() - t
            if r.returncode:
                last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f'exit {r.returncode}'
                report.append((name, 'FAIL', last))
                print(f'[FAIL] {name} ({dt:.0f} s): {last}', flush=True)
                continue
            diffs = []
            for o in outs:
                if o == 'figures':
                    n = len(list((work / 'paper' / 'figures').glob('*.pdf')))
                    diffs += [] if n else ['no figures written']
                    continue
                if o.startswith('stdout:'):
                    new = work / 'results' / 'stdout' / (o[7:] + '.txt')
                    new.write_text(r.stdout)
                    ref = ROOT / 'results' / 'stdout' / (o[7:] + '.txt')
                else:
                    new, ref = work / wpath(o, a.installed), ROOT / o
                    if not ref.exists():                 # examples also at the top of the release
                        ref = ROOT / wpath(o, True)
                if name == 'love_table':
                    ref = ROOT / 'src' / 'earthtides' / 'data' / 'love_PREM.json'
                    if a.installed or not ref.exists():  # the table shipped with the package
                        q = subprocess.run([PY, '-c', 'import earthtides, os; print(os.path.dirname('
                                            'earthtides.__file__))'], cwd=work, env=env,
                                           capture_output=True, text=True)
                        ref = Path(q.stdout.strip()) / 'data' / 'love_PREM.json'
                if not ref.exists():
                    diffs.append(f'no stored reference {ref.relative_to(ROOT)}')
                    continue
                if not new.exists():
                    diffs.append(f'{o} not written')
                    continue
                d = compare_file(new, ref) if name != 'love_table' else compare_love(new, ref)
                diffs += [f'{Path(o).name}: {x}' for x in d[:3]]
            if a.update:                                 # new outputs become the stored results
                for o in outs:
                    if o == 'figures':
                        for f in (work / 'paper' / 'figures').glob('*.pdf'):
                            shutil.copy(f, ROOT / 'paper' / 'figures' / f.name)
                        continue
                    if o.startswith('stdout:'):
                        src, dst = work / 'results' / 'stdout' / (o[7:] + '.txt'), ROOT / 'results' / 'stdout' / (o[7:] + '.txt')
                    else:
                        src, dst = work / wpath(o, a.installed), ROOT / o
                    if name == 'love_table':
                        dst = ROOT / 'src' / 'earthtides' / 'data' / 'love_PREM.json'
                    if src.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src, dst)
            status = 'ok' if not diffs else 'DIFF'
            report.append((name, status, '; '.join(diffs)))
            print(f'[{status:4s}] {name} ({dt:.0f} s){": " + "; ".join(diffs) if diffs else ""}', flush=True)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    n_ok = sum(1 for r in report if r[1] == 'ok')
    print(f'\n{n_ok} of {len(report)} jobs reproduce the stored results '
          f'({sum(1 for r in report if r[1] == "skip")} skipped), {time.time() - t_all:.0f} s')
    if a.keep:
        print(f'outputs kept in {work}')
    sys.exit(0 if all(r[1] in ('ok', 'skip') for r in report) else 1)


def compare_love(new, ref):
    """Love tables: the entries only (the metadata hold dates and runtimes), to 1e-6 relative: the
    K1 entry, at the frequency of the tilt-over mode, differs by ~3e-7 between NumPy builds."""
    import json
    global TOL
    a, b = json.loads(new.read_text()), json.loads(ref.read_text())
    tol, TOL = TOL, 1e-6
    try:
        return compare_text(json.dumps(a['entries']).replace('[', ' ').replace(']', ' '),
                            json.dumps(b['entries']).replace('[', ' ').replace(']', ' '))
    finally:
        TOL = tol


if __name__ == '__main__':
    main()
