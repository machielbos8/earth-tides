# -*- coding: utf-8 -*-
#
# Supplementary figure of paper 1 (PEPI): the solver's viscous core flow on the traced attractors.
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
# 05/11/2026 Machiel Bos
#===============================================================================
"""Quarter of the liquid core (the flow is symmetric about the equator and the same at every longitude):
log10 of |u|^2 of the degrees >= 20 over its median, viscous PREM at 0.9175 cycles per sidereal day (top) and
a neutrally stratified core at 0.9075 (bottom), for delta = 1, 0.5 and 0.25 km; the attractor loop of the
Hamiltonian ray on top; in each panel the energy share within 26 km of the loop over its area share, and the
same for the loop turned by 20 degrees. From saved results. Run from the project root:

    .venv/bin/python paper_pepi/make_figS_core_flow_rays.py   -> paper_pepi/figures/figS_core_flow_rays.pdf

(scripts/core_flow_attractor.py: results/core_flow_attractor_scan.npz (delta 1 km, core K 96, interior in 8
layers N 30, band 12) and _K168_S16N15.npz (delta 0.5, 0.25 km, K 168, 16 layers N 15, band 8, solid_P 0);
shares from their tables and results/core_flow_attractor_scan_control.txt)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper_pepi' / 'figures'
RES = ROOT / 'results'
INK, LOOP = '#0b0b0b', '#5fd3e8'
ETA = 1221.5 / 3480.0
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W2COL = 7.2
ROWS = (('PREM', 0.9175, 'PREM\n0.9175 cpsd'), ('neutral', 0.9075, 'neutral core\n0.9075 cpsd'))
COLS = (('scan', '1', 96, 8), ('K168_S16N15', '0.5', 168, 16), ('K168_S16N15', '0.25', 168, 16))


def shares():
    """(core, f, delta) -> (E/A on the loop, on the turned loop), within 26 km, walls left out."""
    out = {}
    for ln in (RES / 'core_flow_attractor_K168_S16N15.txt').read_text().splitlines()[2:]:
        w = ln.split()
        out[w[1], float(w[0]), w[2]] = (float(w[18]), float(w[19]))
    for ln in (RES / 'core_flow_attractor_scan_control.txt').read_text().splitlines()[2:]:
        k, own, rot = ln.split()
        core, f, d = k.split('_')[:3]
        if d == 'd1':
            out[core, float(f), '1.00'] = (float(own), float(rot))
    return out


def quarter(ax, r, th, E, loop):
    up = th <= np.pi / 2 + 1e-9
    R, T = np.meshgrid(r / 3480e3, th[up], indexing='ij')
    pc = ax.pcolormesh(R * np.sin(T), R * np.cos(T), np.log10(E[:, up] / np.median(E)), vmin=-1.5, vmax=1.5,
                       cmap='magma', shading='gouraud', rasterized=True)
    q = np.abs(loop)                       # folded across the axis and the equator
    jump = np.flatnonzero(np.hypot(*np.diff(q, axis=0).T) > 0.02)
    for a, b in zip(np.r_[0, jump + 1], np.r_[jump + 1, len(q)]):
        ax.plot(q[a:b, 0], q[a:b, 1], color=LOOP, lw=0.5)
    t = np.linspace(0, np.pi / 2, 100)
    for rad in (1.0, ETA):
        ax.plot(rad * np.sin(t), rad * np.cos(t), color=INK, lw=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.axis('off')
    return pc


def main():
    sh = shares()
    data = {c[0]: np.load(RES / f'core_flow_attractor_{c[0]}.npz') for c in COLS}
    fig, axs = plt.subplots(2, 3, figsize=(W2COL * 0.8, 4.2))
    for row, (core, f, lab) in zip(axs, ROWS):
        for ax, (src, d, K, S) in zip(row, COLS):
            k = f'{core}_{f}_d{d}_K{K}_S{S}' + ('_N15' if S == 16 else '')
            D = data[src]
            pc = quarter(ax, D[f'r_{k}'], D[f'th_{k}'], D[f'E_{k}'], D[f'loop_{k}'])
            own, rot = sh[core, f, f'{float(d):.2f}']
            ax.set_title(f'$\\delta$ = {d} km', fontsize=8, loc='left')
            ax.text(0.98, 0.98, f'{own:.2f}\n({rot:.2f})', transform=ax.transAxes, ha='right', va='top',
                    fontsize=7)
        row[0].text(-0.06, 0.45, lab, rotation=90, va='center', ha='right', fontsize=8, transform=row[0].transAxes)
    cb = fig.colorbar(pc, ax=axs, shrink=0.6, pad=0.02)
    cb.set_label(r'$\log_{10}$ of $|\mathbf{u}|^2$ (degrees $\geq20$) over its median')
    fig.savefig(FIG / 'figS_core_flow_rays.pdf', bbox_inches='tight')
    fig.savefig(RES / 'figS_core_flow_rays_preview.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    main()
