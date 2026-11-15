# -*- coding: utf-8 -*-
#
# Figures of paper 1 on the fluid core: inviscid spikes vs viscous core, and the Ekman sub-layers.
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""Fluid-core figures of paper 1 (PEPI), from saved results. Run from the project root:

    .venv/bin/python paper_pepi/make_fig_core.py

fig_core_sweep  h(2) of elastic hydrostatic PREM across the semidiurnal and diurnal bands, inviscid core
                (core K 8, 16, 24) and viscous core (delta = 1 km, core K 96, banded coupling 12; 2026-11-03,
                was delta 4 km K 24): mantle and inner core K 6 / N 24, core N 30, step 0.0005 cpsd
                (results/convergence_viscous_coreK*_N30_d0_s0.0005.txt, ..._coreK96_N30_d1_s0.0005_b12.txt;
                scripts/convergence_viscous_coreK.py)
fig_core_ekman  (a) Chebyshev nodes below the CMB: one outer-core layer, with the 40 km sub-layer, with
                the 40 km and 5 delta sub-layers (EarthModel.viscous_core); (b) the Ekman layer at psi1,
                solver against the local Ekman solution (results/ekman_profile.npz)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import figure_choice as fc          # EARTHTIDES_FIGURE=level: the sweeps of the level figure (suffix _level)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = ROOT / 'paper_pepi' / 'figures'
RES = ROOT / 'results'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#8a8984'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'legend.frameon': False, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W1COL, W2COL = 3.5, 7.2
STYLES = ['-', '--', ':']


DELTA_SWEEP = 1         # [km] delta of the viscous curves (core-K sweeps at 1, 2 and 4 km in results/)
CORE_K = (8, 16, 24)    # core K of the inviscid curves; mantle and inner core K 6 / N 24, core N 30
STEP = 0.0005          # [cpsd] frequency step of the fine sweeps (P1 v2)
CORE_K_VISC = (96,)     # viscous: K 96 only (K 72 within 5e-8 in h2 at delta 1 km, invisible here)
GAP = 0.004            # [cpsd] off-scale values closer than this share one label
BAND_VISC = 12          # banded angular coupling of the K 96 sweep (TidalSolver(band=12))


def sweep():
    # inviscid and viscous core at the same per-layer resolution (Machiel: compare like with like)
    def load(d, Ks, band=None):   # the fine frequency step (0.0005 cpsd) where it exists, else the 0.0025 grid
        b = f'_b{band}' if band else ''
        f = lambda K: next(q for q in (Path(fc.tagged(RES / f'convergence_viscous_coreK{K}_N30_d{d}_s{STEP}{b}.txt')),
                                       Path(fc.tagged(RES / f'convergence_viscous_coreK{K}_N30_d{d}{b}.txt'))) if q.exists())
        return np.vstack([np.genfromtxt(f(K), skip_header=1, usecols=range(9)) for K in Ks])
    inv, vis = load(0, CORE_K), load(DELTA_SWEEP, CORE_K_VISC, BAND_VISC)
    fig, axs = plt.subplots(1, 2, figsize=(W2COL, 2.6))
    for ax, m, tide, ft, lim in ((axs[0], 2, 'M$_2$', 1.92700, (-5.6, -4.3)),
                                 (axs[1], 1, 'O$_1$', 0.92700, (-5.6, -4.3))):
        offs = []
        for d, res, col, lw, lab in ((inv, tuple((K, 30) for K in CORE_K), C2, 0.8, 'inviscid'),
                                     (vis, tuple((K, 30) for K in CORE_K_VISC), C1, 1.6, 'viscous')):
            for st, (K, N) in zip(STYLES, res):
                s = d[(d[:, 0] == m) & (d[:, 2] == K) & (d[:, 3] == N) & (d[:, 1] < 2.4)]
                s = s[np.argsort(s[:, 1])]
                txt = f'{lab}, core $K$={K}'
                ax.plot(s[:, 1], s[:, 5] * 1e4, ls=st, color=col, lw=lw, label=txt)
                off = s[(s[:, 5] * 1e4 < lim[0]) | (s[:, 5] * 1e4 > lim[1])]
                offs += [(x, y * 1e4, col) for x, y in off[:, [1, 5]]]
        # off-scale values: one label per cluster (points closer than GAP, pooled over the curves), the most
        # extreme value, kept inside the axes (2026-11-15: labels overlapped and ran outside the panel)
        x0, x1 = ax.get_xlim()
        for top in (False, True):
            pts = sorted(p for p in offs if (p[1] > lim[1]) == top)
            groups = []
            for p in pts:
                if groups and p[0] - groups[-1][-1][0] < GAP:
                    groups[-1].append(p)
                else:
                    groups.append([p])
            for g in groups:
                x, y, col = max(g, key=lambda q: abs(q[1]))
                right = x > x0 + 0.85 * (x1 - x0)
                ax.annotate(f'{y:.1f}', (x, lim[1] if top else lim[0]),
                            xytext=(-3 if right else 3, -4 if top else 4), textcoords='offset points',
                            ha='right' if right else 'left', va='top' if top else 'bottom', fontsize=6,
                            color=col, annotation_clip=True)
        ax.axvline(ft, color=MUTED, lw=0.6, ls=':')
        ax.text(ft, lim[0] + 0.06, ' ' + tide, va='bottom', ha='left', fontsize=7, color=MUTED)
        ax.set_ylim(*lim)
        ax.set_xlabel('frequency (cycles per sidereal day)')
        ax.set_ylabel(r'$h^{(2)}$ ($10^{-4}$)')
    axs[0].set_title('(a) around M$_2$, $m=2$', loc='left', fontsize=8)
    axs[1].set_title('(b) around O$_1$, $m=1$', loc='left', fontsize=8)
    axs[1].legend(fontsize=6, loc='upper right', ncol=1, borderaxespad=0.8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_core_sweep.pdf')


def ekman():
    from earthtides.model import EarthModel
    md = EarthModel(str(ROOT / 'data' / 'PREM.dat'))
    oc = [i for i in range(md.n_layers) if abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1.0][0]
    cmb, N, delta = md.r1[oc], 30, 1.0e3
    xk = np.cos(np.pi * np.arange(N) / (N - 1))

    def nodes(r0, r1):
        return cmb - (r0 + (xk + 1) / 2 * (r1 - r0))
    sets = [nodes(md.r0[oc], cmb),
            np.concatenate([nodes(md.r0[oc], cmb - 40e3), nodes(cmb - 40e3, cmb)]),
            np.concatenate([nodes(md.r0[oc], cmb - 40e3), nodes(cmb - 40e3, cmb - 5 * delta),
                            nodes(cmb - 5 * delta, cmb)])]
    fig, axs = plt.subplots(1, 2, figsize=(W2COL, 3.0), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]
    for x, z in enumerate(sets):
        z = z[z > 0] / 1e3
        ax.plot(np.full_like(z, x), z, '_', ms=10, mew=1.0, color=C1)
    for d, lab in ((delta, r'$\delta$'), (5 * delta, r'5$\delta$'), (40e3, '40 km')):
        ax.axhline(d / 1e3, color=MUTED, lw=0.6, ls=':')
        ax.text(2.5, d / 1e3, lab, va='center', fontsize=7, color=MUTED)
    ax.set_yscale('log')
    ax.set_ylim(2300, 0.01)
    ax.set_xlim(-0.5, 3.3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['one layer', '+40 km', '+40 km\n+5$\\delta$'])
    ax.set_ylabel('depth below the CMB (km)')
    ax.set_title('(a) Chebyshev nodes, $N=30$ per layer', loc='left', fontsize=8)

    ax = axs[1]
    p = np.load(RES / 'ekman_profile.npz')
    z = p['z'] / p['delta']
    iq = int(p['iq'])
    for k, col in ((1, C1), (2, C3)):
        q, a = p['qm'][:, k], p['am'][k]
        Q = q[iq]
        lab = f"{np.degrees(p['th'][k]):.0f}$^\\circ$"
        ax.plot(z, (q / Q).real, '-', color=col, label=f'Re, colatitude {lab}')
        ax.plot(z, (q / Q).imag, '--', color=col, label=f'Im, colatitude {lab}')
        ax.plot(z, (a / Q).real, '-', color=INK, lw=0.5)
        ax.plot(z, (a / Q).imag, '--', color=INK, lw=0.5)
    nz = p['nodes'] / p['delta']
    ax.plot(nz[nz < 12], np.full((nz < 12).sum(), -0.25), '|', color=MUTED, ms=5)
    ax.set_xlim(0, 12)
    ax.set_xlabel(r'depth below the CMB / $\delta$')
    ax.set_ylabel('flow relative to the mantle / interior value')
    ax.set_title(r'(b) Ekman layer at $\psi_1$, $\delta$ = 1 km', loc='left', fontsize=8)
    ax.plot([], [], color=INK, lw=0.5, label='Ekman solution')
    ax.legend(fontsize=6, loc='center right', bbox_to_anchor=(1.0, 0.62))   # clear of the Im curves near 0
    fig.tight_layout()
    fig.savefig(FIG / 'fig_core_ekman.pdf')


if __name__ == '__main__':
    sweep()
    if (RES / 'ekman_profile.npz').exists():
        ekman()
