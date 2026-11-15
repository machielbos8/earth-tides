# -*- coding: utf-8 -*-
#
# Fig. 2 of P1 (PEPI): h0 and k through the diurnal band, with Dehant et al. (1999), on the level figure.
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""Fig. 2 (fig_fcn.pdf): h0 and k of elastic, hydrostatic PREM across the diurnal band, with the values of
Dehant et al. (1999, tables 11/12). Replaces the figure of paper/make_figures.py (fig_fcn, data of
scripts/run_diurnal.py: first-order figure, core K 3, 90 frequencies, step 5e-4 cpsd near 0.9975).
Here: the figure of EARTHTIDES_FIGURE (default 'level'), slipping core with the term of a non-level wall
(solver default), K 4, N 24, as the Love tables of the paper; a fine step (2.5e-5 cpsd) from 0.996 to 0.999
cpsd, where the free inner core nutation shows, 5e-4 elsewhere below K1, 1e-4 between K1 and the FCN,
2.5e-4 above it. The FCN from a pole fit to h0 (1.0012 .. 1.0055 cpsd); the FICN from the largest |dh0| in
the fine window (the curve is broken there). K1 itself (the tilt-over mode, singular) is left out.

    .venv/bin/python paper_pepi/make_fig_fcn.py [NPROC] [--replot]
    ->  paper_pepi/figures/fig_fcn.pdf, results/figdata_fcn[_level].npz"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('EARTHTIDES_FIGURE', 'level')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from figure_choice import FIGURE, SUF
from provenance import Run

CACHE = ROOT / 'results' / f'figdata_fcn{SUF}.npz'
DDW = {'O1': (0.92700, 0.59553, 0.29335), 'P1': (0.99454, 0.57411, 0.28249), 'K1': (1.00000, 0.51400, 0.25205),
       'psi1': (1.00273, 0.93394, 0.46477), 'phi1': (1.00546, 0.65422, 0.32308)}
SWEEP = np.unique(np.round(np.concatenate([
    np.arange(0.985, 0.996, 5e-4), np.arange(0.996, 0.999 + 1e-9, 2.5e-5), np.arange(0.9995, 0.99951, 5e-4),
    np.arange(1.0005, 1.00191, 1e-4), np.arange(1.00225, 1.0121, 2.5e-4)]), 7))
FIT = np.array([1.0012, 1.0015, 1.0018, 1.0020, 1.0024, 1.0027, 1.0032, 1.0040, 1.0055])
_G = {}


def solve(x):
    from earthtides.model import EarthModel
    from earthtides.geometry import OMEGA_EARTH, figure, polar_equatorial_radius
    from earthtides.solver import TidalSolver
    from earthtides.iers import iers_fit
    if not _G:
        md = EarthModel(str(ROOT / 'data' / 'PREM.dat'))
        g = figure(md, FIGURE)
        _G.update(md=md, g=g, conv=(md.a / polar_equatorial_radius(g, md)[1]) ** 5, OM=OMEGA_EARTH)
    S = TidalSolver(_G['md'], _G['g'], m=1, K=4, N=24, omega_f=0.0, coriolis=_G['OM'], omega_t=x * _G['OM'],
                    omega_rot=_G['OM'], wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle')
    r = S.solve()
    f = iers_fit(S, _G['g'])
    return x, float(f['h0'].real), float((r['kk'] * _G['conv']).real)


def compute(run, nproc):
    with Pool(nproc) as p:
        R = p.map(solve, list(SWEEP) + list(FIT), chunksize=4)
    D = {x: (h, k) for x, h, k in R}
    h = np.array([D[x][0] for x in SWEEP])
    k = np.array([D[x][1] for x in SWEEP])
    hf = np.array([D[x][0] for x in FIT])
    fcn = least_squares(lambda q: q[0] + q[1] / (q[2] - FIT) - hf, [0.6, -1e-3, 1.00218]).x[2]
    d = dict(sweep=SWEEP, h0=h, k0=k, fcn=fcn, ddw_f=np.array([v[0] for v in DDW.values()]),
             ddw_h0=np.array([v[1] for v in DDW.values()]), ddw_k0=np.array([v[2] for v in DDW.values()]))
    run.savez(str(CACHE.relative_to(ROOT)), **d)
    return d


def ficn_of(F):
    """The free inner core nutation: midpoint of the largest jump of h0 in the fine window (an undamped pole of
    the elastic Earth with an inviscid core; the curve is broken there as at the FCN)."""
    x, h = F['sweep'], F['h0']
    w = np.where((x >= 0.996) & (x <= 0.999))[0]
    j = w[int(np.argmax(np.abs(np.diff(h[w]))))]
    return 0.5 * (x[j] + x[j + 1])


def plot(F):
    sys.path.insert(0, str(ROOT / 'paper'))
    import make_figures as mf                 # style and colours of the paper figures
    plt = mf.plt
    x, fcn = F['sweep'], float(F['fcn'])
    ficn = ficn_of(F)
    fig, axs = plt.subplots(1, 2, figsize=(mf.W2COL, 2.4))
    for ax, key, dkey, lab in ((axs[0], 'h0', 'ddw_h0', '$h^{(0)}$'), (axs[1], 'k0', 'ddw_k0', '$k$')):
        y = F[key]
        for i, seg in enumerate((x < ficn, (x > ficn) & (x < 1.0), (x > 1.0) & (x < fcn), x > fcn)):
            ax.plot(x[seg], y[seg], color=mf.C1, lw=1.2, label='this paper' if i == 0 else None)
        ax.plot(F['ddw_f'], F[dkey], 'o', mfc='none', mec=mf.C2, ms=5, label='Dehant et al. (1999)')
        ax.axvline(fcn, color=mf.MUTED, lw=0.5, ls='--')
        ax.axvline(1.0, color=mf.MUTED, lw=0.5, ls=':')
        ax.axvline(ficn, color=mf.MUTED, lw=0.5, ls='--')
        ax.set_xlim(0.985, 1.012)
        ax.set_xlabel('frequency (cycles per sidereal day)')
        ax.set_ylabel(lab)
    axs[0].set_ylim(0.3, 1.2)
    axs[1].set_ylim(0.15, 0.6)
    axs[0].text(fcn + 0.0004, 1.12, 'FCN', color=mf.MUTED, fontsize=7)
    axs[0].text(1.0004, 1.12, 'K$_1$', color=mf.MUTED, fontsize=7)
    axs[0].text(ficn - 0.0004, 1.12, 'FICN', color=mf.MUTED, fontsize=7, ha='right')
    axs[0].legend(fontsize=6.5, loc='lower left')
    axs[0].set_title('(a)', loc='left', fontsize=8)
    axs[1].set_title('(b)', loc='left', fontsize=8)
    fig.tight_layout()
    out = ROOT / 'paper_pepi' / 'figures' / 'fig_fcn.pdf'
    fig.savefig(out)
    return out


def main():
    os.chdir(ROOT)
    nproc = int(next((a for a in sys.argv[1:] if a.isdigit()), 4))
    with Run(f'Fig. 2 of P1: h0 and k through the diurnal band ({FIGURE} figure, slipping core K 4 N 24)',
             params=dict(figure=FIGURE, n=len(SWEEP), fine_step=2.5e-5, K=4, N=24), inputs=['data/PREM.dat']) as run:
        F = dict(np.load(CACHE)) if ('--replot' in sys.argv and CACHE.exists()) else compute(run, nproc)
        out = plot(F)
        run.output(str(out.relative_to(ROOT)))
        fine = (F['sweep'] >= 0.996) & (F['sweep'] <= 0.999)
        xf, hf = F['sweep'][fine], F['h0'][fine]
        j = int(np.argmax(np.abs(np.diff(hf))))
        print(f'{len(F["sweep"])} frequencies; FCN {float(F["fcn"]):.6f} cpsd = 1 + 1/{1 / (float(F["fcn"]) - 1):.1f} sd; '
              f'largest jump of h0 in 0.996-0.999 between {xf[j]:.5f} and {xf[j + 1]:.5f} '
              f'({hf[j]:.5f} -> {hf[j + 1]:.5f})')


if __name__ == '__main__':
    main()
