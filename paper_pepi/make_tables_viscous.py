# -*- coding: utf-8 -*-
#
# Tables of paper 1 (PEPI) with the viscous outer core (EarthModel.viscous_core): the latitude
# table (tab:latitude) and the comparison with DDW (tab:ddw).
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Tables of paper 1 (PEPI) with the viscous outer core. Elastic, hydrostatic PREM; the liquid core
has a Newtonian viscosity (mu = i omega eta) with Ekman boundary layers of 40 km at the ICB and CMB
(EarthModel.viscous_core); mantle frame, IERS conventions, consistent reference potential. Each value
is computed for Ekman thicknesses delta = 2, 1, 0.5 km and extrapolated linearly to delta -> 0 from
1 and 0.5 km (the viscous coupling is linear in delta); the 2 km run checks the extrapolation.
K = 7, N = 30. K1 is the mean of 1 -+ 1e-5 cpsd (tilt-over mode at 1 cpsd).
Run from the project root:  .venv/bin/python paper_pepi/make_tables_viscous.py
-> paper_pepi/table_latitude_viscous.tex, paper_pepi/table_ddw_viscous.tex (NOT used by main.tex:
the extrapolation fails near the FCN resonance, see lab notes), results/paper1_tables_viscous.txt
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
# tex tables go to results/regen/ (for comparison) unless --paper: the paper's tables may carry
# values from other runs (lab notes 2026-11-08)
OUT = ROOT / 'paper_pepi' if '--paper' in sys.argv else ROOT / 'results' / 'regen'
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

DELTAS = (2e3, 1e3, 5e2)
K, N = 7, 30
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
# DDW1999 Tables 11, 12 (elastic hydrostatic PREM): h0, h2, l0, l1, l2, k0; Mf from Table 11
DDW = {'zonal': (0, 0.07300, (0.59892, -0.00047, 0.08304, None, 0.00008, 0.29473)),
       'O$_1$': (1, 0.92700, (0.59553, -0.00048, 0.08298, 0.00070, 0.00008, 0.29335)),
       'P$_1$': (1, 0.99454, (0.57411, -0.00053, 0.08367, 0.00070, 0.00009, 0.28249)),
       'K$_1$': (1, 1.00000, (0.51400, -0.00065, 0.08558, 0.00060, 0.00011, 0.25205)),
       '$\\psi_1$': (1, 1.00273, (0.93394, 0.00024, 0.07227, 0.00130, -0.00004, 0.46477)),
       '$\\phi_1$': (1, 1.00546, (0.65422, -0.00036, 0.08113, 0.00084, 0.00006, 0.32308)),
       'OO$_1$': (1, 1.07300, (0.60092, -0.00047, 0.08283, 0.00077, 0.00008, 0.29607)),
       'M$_2$': (2, 1.92700, (0.60175, -0.00047, 0.08359, 0.00100, 0.00008, 0.29751))}
LAT_WAVES = (('Mf', 0, 0.07300), ('O$_1$', 1, 0.92700), ('M$_2$', 2, 1.92700))
LAT_CASES = (('flattening only', 1.0, 0.0), ('Coriolis force only', 0.0, OMEGA_EARTH),
             ('both', 1.0, OMEGA_EARTH))


def solve(job):
    m, sig, delta, scale, cor = job
    w = sig * OMEGA_EARTH
    md = EarthModel(ROOT / 'data/PREM.dat').viscous_core(w, delta=delta)
    import figure_choice as fc
    g = fc.base(md, scale=scale)
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=cor, omega_t=w,
                    omega_rot=OMEGA_EARTH * np.sqrt(scale), wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5      # DDW normalisation of k
    print('m=%d %.5f delta %.1f km scale %.0f cor %.0e  h0 %+.6f h2 %+.6f' % (
        m, sig, delta / 1e3, scale, cor, f['h0'].real, f['h2'].real), flush=True)
    return job, {k: complex(f[k]) for k in KEYS}


def extrap(v):
    """v: {delta: dict}; linear to delta -> 0 from 1 and 0.5 km; check: 2 km."""
    a, b, c = v[2e3], v[1e3], v[5e2]
    e = {k: 2 * c[k] - b[k] for k in KEYS}
    chk = {k: abs((2 * b[k] - a[k]) - e[k]) for k in KEYS}      # same extrapolation from 2, 1 km
    return e, chk


def _main(run):
    jobs = set()
    for m, s, _ in DDW.values():
        for f in ((s - 1e-5, s + 1e-5) if s == 1.0 else (s,)):
            for d in DELTAS:
                jobs.add((m, f, d, 1.0, OMEGA_EARTH))
    for _, sc, cor in LAT_CASES:
        for _, m, s in LAT_WAVES:
            for d in DELTAS:
                jobs.add((m, s, d, sc, cor))
    jobs = sorted(jobs, key=lambda j: (-j[2], j[0], j[1]))
    with Pool(2, maxtasksperchild=1) as p:
        R = dict(p.map(solve, jobs, chunksize=1))

    def value(m, s, sc=1.0, cor=OMEGA_EARTH):
        fs = (s - 1e-5, s + 1e-5) if s == 1.0 else (s,)
        v = {d: {k: np.mean([R[(m, f, d, sc, cor)][k] for f in fs]) for k in KEYS} for d in DELTAS}
        return extrap(v) + (v,)

    log = ['delta -> 0 extrapolation from 1 and 0.5 km; chk = |extrapolation from 2, 1 km - that from 1, 0.5 km|',
           '%-22s %-10s ' % ('', '') + ' '.join('%10s' % k for k in KEYS)]
    fmt = lambda x: f'{round(x, 5) + 0.0:.5f}' if x >= -0.000005 else f'$-{abs(round(x, 5)):.5f}$'
    # --- DDW table
    rows = []
    for name, (m, s, ref) in DDW.items():
        e, chk, v = value(m, s)
        log.append('%-22s %-10s ' % (name, 'value') + ' '.join('%+12.9f' % e[k].real for k in KEYS))
        log.append('%-22s %-10s ' % ('', 'Im') + ' '.join('%+10.1e' % e[k].imag for k in KEYS))
        log.append('%-22s %-10s ' % ('', 'chk') + ' '.join('%10.1e' % chk[k] for k in KEYS))
        for d in DELTAS:
            log.append('%-22s %-10s ' % ('', 'd=%.1fkm' % (d / 1e3)) + ' '.join('%+10.6f' % v[d][k].real for k in KEYS))
        lab = 'zonal' if name == 'zonal' else name
        rows.append(f'{lab} & {s:.5f} & ' + ' & '.join(fmt(e[k].real) for k in KEYS) + r' \\')
        rows.append(' & DDW & ' + ' & '.join('' if x is None else fmt(x) for x in ref) + r' \\')
    rows[0] = rows[0].replace('0.07300', '0.0730')
    ddw_tex = r"""\begin{table*}
\caption{Degree-2 Love numbers of elastic, hydrostatic PREM in the definition
of \citet{mathews1995}: this paper (first line) and \citet[][DDW,
tables~11 and 12]{dehant1999} (second line). Frequencies in cycles per sidereal
day; the zonal row is for Mf. The small terms $h'$ and $l'$, which DDW give in other normalizations and,
for $l'$, another frame, are compared in the Supplementary Material
(Sect.~\ref{S-sec:offsets}).}
\label{tab:ddw}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}
\toprule
tide & frequency & $h^{(0)}$ & $h^{(2)}$ & $l^{(0)}$ & $l^{(1)}$ & $l^{(2)}$ & $k$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    import figure_choice as fc
    run.write_text(str(OUT / f'table_ddw_viscous{fc.SUF}.tex'), ddw_tex)
    # --- latitude table
    LK = ('h2', 'l2', 'l1')
    lat = {}
    for case, sc, cor in LAT_CASES:
        for w, m, s in LAT_WAVES:
            e, chk, v = value(m, s, sc, cor)
            lat[case, w] = e
            log.append('%-22s %-10s ' % (case, w.replace('$', '')) + ' '.join('%+10.6f' % e[k].real for k in KEYS)
                       + '   chk ' + ' '.join('%.0e' % chk[k] for k in LK))
    head = ' & '.join(f'\\multicolumn{{3}}{{c}}{{{w}}}' for w, _, _ in LAT_WAVES)
    sub = ' & '.join(['$h^{(2)}$ & $l^{(2)}$ & $l^{(1)}$'] * 3)
    f5 = lambda x: f'${round(x, 5) + 0.0:.5f}$'
    lrows = [f'{case} & ' + ' & '.join(f5(lat[case, w][k].real) for w, _, _ in LAT_WAVES for k in LK) + r' \\'
             for case, _, _ in LAT_CASES]
    lat_tex = r"""\begin{table*}
\caption{Origin of the latitude dependence: $h^{(2)}$, $l^{(2)}$ and $l^{(1)}$ of elastic,
hydrostatic PREM (fluid core, mantle frame) with the flattening alone (no Coriolis force), the
Coriolis force alone (spherical Earth) and both.}
\label{tab:latitude}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & """ + head + r""" \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
 & """ + sub + r""" \\
\midrule
""" + '\n'.join(lrows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    run.write_text(str(OUT / f'table_latitude_viscous{fc.SUF}.tex'), lat_tex)
    run.write_text(str(ROOT / 'results' / f'paper1_tables_viscous{fc.SUF}.txt'), '\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    from provenance import Run
    with Run('Viscous-core DDW and latitude tables, delta -> 0 (P1 v2 Table S3)', params=dict(out=str(OUT)), inputs=['data/PREM.dat']) as run_:
        _main(run_)
