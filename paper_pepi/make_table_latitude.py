# -*- coding: utf-8 -*-
"""Table of paper 1 (PEPI): origin of the latitude dependence. Elastic, hydrostatic PREM with a
true fluid core, mantle frame, IERS conventions, at Mf, O1 and M2 for three cases: flattening
only (no Coriolis force), Coriolis force only (spherical Earth) and both. Run from the project
root:  .venv/bin/python paper_pepi/make_table_latitude.py  -> paper_pepi/table_latitude.tex
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
# tex tables go to results/regen/ (for comparison) unless --paper: the paper's tables may carry
# values from other runs (lab notes 2026-11-08)
OUT = ROOT / 'paper_pepi' if '--paper' in sys.argv else ROOT / 'results' / 'regen'
# --figure level: the flattened cases on the level figure (rotation rate Omega sqrt(scale)); suffix _level
FIGURE = sys.argv[sys.argv.index('--figure') + 1] if '--figure' in sys.argv else 'first-order'   # or 'level'
SUF = '' if FIGURE == 'first-order' else '_' + FIGURE
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, figure
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

md = EarthModel(ROOT / 'data/PREM.dat')
WAVES = (('Mf', 0, 0.07300), ('O$_1$', 1, 0.92700), ('M$_2$', 2, 1.92700))
CASES = (('flattening only', 1.0, 0.0), ('Coriolis force only', 0.0, OMEGA_EARTH),
         ('both', 1.0, OMEGA_EARTH))
KEYS = ('h2', 'l2', 'l1')


def solve(m, sigma, scale, cor):
    g = figure(md, FIGURE, scale=scale)
    S = TidalSolver(md, g, m=m, K=7, N=40, omega_f=0.0, coriolis=cor, omega_t=sigma * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH * np.sqrt(scale), wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    return iers_fit(S, g)


def _main(run):
    R = {}
    for case, scale, cor in CASES:
        for w, m, s in WAVES:
            R[case, w] = solve(m, s, scale, cor)
            print(case, w, ' '.join(f'{k} {R[case, w][k].real:+.5f}' for k in KEYS), flush=True)
    def fmt(x):
        x = round(x, 5) + 0.0
        return f'${x:.5f}$'
    head = ' & '.join(f'\\multicolumn{{3}}{{c}}{{{w}}}' for w, _, _ in WAVES)
    sub = ' & '.join(['$h^{(2)}$ & $l^{(2)}$ & $l^{(1)}$'] * 3)
    rows = [f'{case} & ' + ' & '.join(fmt(R[case, w][k].real) for w, _, _ in WAVES for k in KEYS) + r' \\'
            for case, _, _ in CASES]
    tex = r"""\begin{table*}
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
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    run.write_text(str(OUT / f'table_latitude{SUF}.tex'), tex)
    run.write_text(str(ROOT / 'results' / f'paper1_table_latitude{SUF}.txt'), 
        '\n'.join(f'{c:22s} {w:8s} ' + ' '.join(f'{k} {R[c, w][k].real:+.6f}' for k in ('h0', 'h2', 'l0', 'l2', 'l1'))
                  for c, _, _ in CASES for w, _, _ in WAVES) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    from provenance import Run
    with Run('Table of the latitude dependence, inviscid core (P1 v2 Table S8)', params=dict(out=str(OUT)), inputs=['data/PREM.dat']) as run_:
        _main(run_)
