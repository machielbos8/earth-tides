# -*- coding: utf-8 -*-
#
# Figure of paper 1 (PEPI): the inertial waves of the solver in a plausible core, neutral with stable
# layers below the core-mantle boundary and above the inner core, against PREM's (Fig. rays_buoyancy).
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
# 06/11/2026 Machiel Bos
#===============================================================================
"""(a, b) As Fig. rays_buoyancy, for a neutrally stratified core with a stable layer of 140 km below the CMB
and an F-layer of 150 km above the ICB, both N = Omega (convergence_stratification.layered), viscous core
delta = 0.25 km: log10 of |u|^2 of the degrees >= 20 over its median; the attractor loop traced with Hamilton's
equations where the ray closes (M2); the outline of the regions where no wave can travel (dashed) and the edges
of the stable layers (dotted). (c) N^2 of this core and of PREM's (grey). From saved results:

    .venv/bin/python paper_pepi/make_fig_layered_core.py   -> paper_pepi/figures/fig_layered_core.pdf

(scripts/core_flow_attractor.py: results/core_flow_attractor_layered.npz, cases f:layered:0.25:168:16:15,
band 8, solid_P 0, --full-maps; scripts/rays_disc_grid.py [--core layered]: results/rays_disc_grid[_layered].npz)
"""
import numpy as np
import matplotlib.pyplot as plt

from make_fig_rays_buoyancy import quarter, FIG, RES, W2COL, C1, C3, INK, MUTED, EDGE

CASES = ((0.9175, 'layered_0.9175_d0.25_K168_S16_N15', '(a) 0.9175 cycles per sidereal day'),
         (1.92700, 'layered_1.927_d0.25_K168_S16_N15', '(b) M$_2$'))
TOP, FL = 140.0, 150.0           # km, the stable layers of convergence_stratification.layered


def main():
    g = np.load(RES / 'rays_disc_grid_layered.npz')
    p = np.load(RES / 'rays_disc_grid.npz')
    s, eta, rc = g['s'], float(g['eta']), float(g['rc'])
    d = np.load(RES / 'core_flow_attractor_layered.npz')
    fig = plt.figure(figsize=(W2COL, 3.1))
    axs = [fig.add_axes([0.0, 0.2, 0.33, 0.72]), fig.add_axes([0.35, 0.2, 0.33, 0.72]),
           fig.add_axes([0.80, 0.2, 0.19, 0.72])]
    t = np.linspace(0, np.pi / 2, 200)
    for ax, (f, k, title) in zip(axs[:2], CASES):
        loop = d[f'loop_{k}']
        pc = quarter(ax, d[f'r_{k}'], d[f'th_{k}'], d[f'E_{k}'], loop if len(loop) > 1 else np.zeros((0, 2)),
                     s, g[f'disc_{f}'], eta, None)
        for rad in (1 - TOP / (rc / 1e3), eta + FL / (rc / 1e3)):
            ax.plot(rad * np.sin(t), rad * np.cos(t), color=EDGE, lw=0.6, ls=':')
        ax.set_title(title, loc='left', fontsize=8)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.12, 0.1, 0.44, 0.03]), orientation='horizontal')
    cb.set_label(r'$\log_{10}$ of $|\mathbf{u}|^2$ (degrees $\geq20$) over its median')
    ax = axs[2]
    ax.plot(p['n2'], p['x'] * rc / 1e3, color=MUTED, lw=0.8, label='PREM')
    ax.plot(g['n2'], g['x'] * rc / 1e3, color=C1, label='this core')
    ax.axvline(0, color=MUTED, lw=0.6)
    for f, col, nm in ((0.9175, INK, '0.9175'), (1.92700, C3, 'M$_2$')):
        ax.axvline((f / 2) ** 2, color=col, lw=0.8, ls=':', label=nm + r': $\hat f^2$')
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(g['x'].min() * rc / 1e3, rc / 1e3)
    ax.set_xlabel(r'$N^2/(2\Omega)^2$')
    ax.set_ylabel('radius (km)')
    ax.set_title('(c) liquid core', loc='left', fontsize=8)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.32, -0.2), handlelength=2.2, borderaxespad=0.0,
              ncol=2)
    fig.savefig(FIG / 'fig_layered_core.pdf', bbox_inches='tight')
    fig.savefig(RES / 'fig_layered_core_preview.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    main()
