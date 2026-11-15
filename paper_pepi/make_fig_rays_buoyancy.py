# -*- coding: utf-8 -*-
#
# Figure of paper 1 (PEPI): the inertial waves of the solver in PREM's liquid core, with the traced rays and the
# regions where no wave can travel.
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
# 03/11/2026 Machiel Bos
#===============================================================================
"""(a, b) Quarter of the liquid core of PREM (the flow is symmetric about the equator and the same at every
longitude) at 0.9175 cycles per sidereal day and at M2, viscous core delta = 0.25 km: log10 of |u|^2 of the
spherical-harmonic degrees >= 20 (the beams) over its median; the ray traced with Hamilton's equations (its
last segments: the attractor at 0.9175); the outline of the region where no wave of that frequency can travel.
(c) N^2 of PREM's core with the thresholds of the two frequencies (on the axis n^2 = fh^2, in the equatorial
plane n^2 = fh^2 - 1). From saved results. Run from the project root:

    .venv/bin/python paper_pepi/make_fig_rays_buoyancy.py   -> paper_pepi/figures/fig_rays_buoyancy.pdf

(scripts/core_flow_attractor.py: results/core_flow_attractor_K168_S16N15.npz and _M2.npz, core K 168, interior
in 16 layers N 15, band 8, solid_P 0, maps at every grid point; scripts/rays_disc_grid.py: results/rays_disc_grid.npz)
Until 2026-11-05 this figure showed the rays alone (0.907 cpsd and M2, with and without N^2; git history).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper_pepi' / 'figures'
RES = ROOT / 'results'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, LOOP, EDGE = '#0b0b0b', '#8a8984', '#5fd3e8', '#ffffff'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'legend.frameon': False, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W2COL = 7.2
# (frequency, file, key, title, ray segments drawn: the attractor loop at 0.9175, the last 6 at M2, which has none)
CASES = ((0.9175, 'core_flow_attractor_K168_S16N15.npz', 'PREM_0.9175_d0.25_K168_S16_N15',
          '(a) 0.9175 cycles per sidereal day', None),
         (1.92700, 'core_flow_attractor_M2.npz', 'PREM_1.927_d0.25_K168_S16_N15', '(b) M$_2$', 6))


def quarter(ax, r, th, E, path, s, D, eta, nseg=None):
    up = th <= np.pi / 2 + 1e-9
    R, T = np.meshgrid(r / 3480e3, th[up], indexing='ij')
    pc = ax.pcolormesh(R * np.sin(T), R * np.cos(T), np.log10(E[:, up] / np.median(E)), vmin=-1.5, vmax=1.5,
                       cmap='magma', shading='gouraud', rasterized=True)
    if nseg is not None:                   # the last nseg segments (wall points: r = 1 or eta)
        rr = np.hypot(*path.T)
        w = np.flatnonzero((np.abs(rr - 1) < 1e-6) | (np.abs(rr - eta) < 1e-6))
        path = path[w[-nseg - 1]:]
    q = np.abs(path)                       # folded across the axis and the equator
    jump = np.flatnonzero(np.hypot(*np.diff(q, axis=0).T) > 0.02)
    for a, b in zip(np.r_[0, jump + 1], np.r_[jump + 1, len(q)]):
        ax.plot(q[a:b, 0], q[a:b, 1], color=LOOP, lw=0.6)
    ax.contour(s, s, D, levels=[0.0], colors=[EDGE], linewidths=0.9, linestyles='--')
    t = np.linspace(0, np.pi / 2, 200)
    for rad in (1.0, eta):
        ax.plot(rad * np.sin(t), rad * np.cos(t), color=INK, lw=0.7)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.axis('off')
    return pc


def main():
    g = np.load(RES / 'rays_disc_grid.npz')
    s, eta = g['s'], float(g['eta'])
    fig = plt.figure(figsize=(W2COL, 3.1))
    axs = [fig.add_axes([0.0, 0.2, 0.33, 0.72]), fig.add_axes([0.35, 0.2, 0.33, 0.72]),
           fig.add_axes([0.80, 0.2, 0.19, 0.72])]
    for ax, (f, src, k, title, nseg) in zip(axs[:2], CASES):
        d = np.load(RES / src)
        pc = quarter(ax, d[f'r_{k}'], d[f'th_{k}'], d[f'E_{k}'], d[f'loop_{k}'], s, g[f'disc_{f}'], eta, nseg)
        ax.set_title(title, loc='left', fontsize=8)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.12, 0.1, 0.44, 0.03]), orientation='horizontal')
    cb.set_label(r'$\log_{10}$ of $|\mathbf{u}|^2$ (degrees $\geq20$) over its median')
    ax = axs[2]
    r = g['x'] * float(g['rc']) / 1e3
    ax.plot(g['n2'], r, color=C1)
    ax.axvline(0, color=MUTED, lw=0.6)
    for f, col, nm in ((0.9175, INK, '0.9175'), (1.92700, C3, 'M$_2$')):
        fh2 = (f / 2) ** 2
        ax.axvline(fh2 - 1, color=col, lw=0.8, ls='--', label=nm + r': $\hat f^2-1$')
        if fh2 < 0.4:
            ax.axvline(fh2, color=col, lw=0.8, ls=':', label=nm + r': $\hat f^2$')
    ax.set_xlim(-1.5, 0.35)
    ax.set_ylim(r.min(), r.max())
    ax.set_xlabel(r'$N^2/(2\Omega)^2$')
    ax.set_ylabel('radius (km)')
    ax.set_title('(c) PREM, liquid core', loc='left', fontsize=8)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.32, -0.2), handlelength=2.2, borderaxespad=0.0)
    fig.savefig(FIG / 'fig_rays_buoyancy.pdf', bbox_inches='tight')
    fig.savefig(RES / 'fig_rays_buoyancy_preview.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    main()
