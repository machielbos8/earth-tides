# -*- coding: utf-8 -*-
#
# Fig. S2 of the P1 supplement (PEPI): Q and the shear modulus from the seismic to the tidal periods.
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
"""How the shear modulus of PREM, defined at 1 s, is carried to tidal periods (absorption band of Anderson &
Minster 1979 in the form of Wahr & Bergen 1986, alpha = 0.15, T_m = 200 s):
  (a) Q_mu relative to its seismic value: constant from 1 s to T_m, Q ~ omega^alpha beyond (both models);
  (b) Re delta mu / mu for Q_mu = 312 (PREM lower mantle):
      this paper: dispersion from 1 s (constant-Q band 1-200 s, then the power law);
      Mathews et al. (1997) / IERS: the PREM modulus taken at 200 s, power law only.
Beyond T_m the curves are model.EarthModel.dmu_factor (checked here); below T_m the constant-Q law
(2/pi) ln(omega/omega_0) / Q of the same function.

    .venv/bin/python paper_pepi/make_fig_dispersion.py
    ->  paper_pepi/figures/figS_dispersion.pdf"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper'))
import numpy as np

from earthtides.model import EarthModel
from provenance import Run

OUT = ROOT / 'paper_pepi' / 'figures' / 'figS_dispersion.pdf'
ALPHA, TM, Q = 0.15, 200.0, 312.0
TIDES = (('M$_2$', 12.42 * 3600), ('O$_1$', 25.82 * 3600), ('Mf', 13.66 * 86400))


def dmu(T, T0):
    """Re delta mu / mu at period T [s] for reference period T0 (T0 = TM: the IERS assumption)."""
    T = np.atleast_1d(T).astype(float)
    w, w0, wm = 2 * np.pi / T, 2 * np.pi / T0, 2 * np.pi / TM
    band = 2 / np.pi * np.log(w / w0) / Q                                  # constant Q, T0 <= T <= TM
    x = (wm / w)**ALPHA
    power = (2 / np.pi * np.log(wm / w0) + (1 - x) / np.tan(ALPHA * np.pi / 2)) / Q
    return np.where(T <= TM, band, power)


def check():
    ns = SimpleNamespace(Qmu=[Q], n_layers=1)
    for T in (TIDES[0][1], TIDES[2][1]):
        for T0 in (1.0, TM):
            ref = EarthModel.dmu_factor(ns, 2 * np.pi / T, alpha=ALPHA, Tm=TM, T0=T0)[0].real
            assert abs(ref - dmu(T, T0)[0]) < 1e-15, (T, T0)


def plot():
    import make_figures as mf
    plt = mf.plt
    T = np.logspace(0, np.log10(3e6), 400)
    fig, (a, b) = plt.subplots(2, 1, figsize=(mf.W1COL, 3.6), sharex=True)
    q = np.where(T <= TM, 1.0, (T / TM)**(-ALPHA))
    a.plot(T, q, color=mf.INK, lw=1.2)
    a.set_ylabel(r'$Q_\mu\,/\,Q_\mu$(1 s)')
    a.set_ylim(0, 1.15)
    a.text(1.4, 1.03, 'constant $Q$', fontsize=6.5)
    a.text(1.5e3, 0.42, r'$Q\propto\omega^{0.15}$', fontsize=6.5)
    ours, iers = 100 * dmu(T, 1.0), 100 * dmu(T, TM)
    b.plot(T, ours, color=mf.C1, lw=1.4, label='this paper: PREM at 1 s')
    b.plot(T[T >= TM], iers[T >= TM], color=mf.C2, lw=1.4, ls='--', label='Conventions: PREM at 200 s')
    b.plot([1.0], [0.0], 'o', color=mf.C1, ms=4)
    b.plot([TM], [0.0], 'o', color=mf.C2, ms=4, mfc='white')
    b.set_ylabel(r'$\Delta\mu/\mu$ (%), $Q_\mu=312$')
    b.set_xlabel('period (s)')
    b.legend(fontsize=6, loc='lower left')
    for ax in (a, b):
        ax.set_xscale('log')
        ax.axvline(TM, color=mf.MUTED, lw=0.5, ls=':')
        for name, t in TIDES:
            ax.axvline(t, color=mf.MUTED, lw=0.5)
    for name, t in TIDES:
        a.text(t * 1.15, 0.08, name, fontsize=6.5, color=mf.MUTED, rotation=90, va='bottom')
    a.text(TM * 1.15, 0.08, '200 s', fontsize=6.5, color=mf.MUTED, rotation=90, va='bottom')
    a.set_title('(a)', loc='left', fontsize=8)
    b.set_title('(b)', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT)
    for name, t in TIDES:
        print(f'{name:8s} dmu/mu: from 1 s {100 * dmu(t, 1.0)[0]:+.2f} %, from 200 s {100 * dmu(t, TM)[0]:+.2f} %')
    return OUT


def main():
    with Run('Fig. S2 of the P1 supplement: Q and shear-modulus dispersion, PREM at 1 s vs at 200 s',
             params=dict(alpha=ALPHA, Tm=TM, Q=Q), inputs=[]) as run:
        check()
        run.output(str(plot().relative_to(ROOT)))


if __name__ == '__main__':
    main()
