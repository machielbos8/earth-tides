# -*- coding: utf-8 -*-
#
# Supplementary figure of paper 1 (PEPI): the solver's beams in flattened PREM and in a sphere.
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
"""Energy of the degrees >= 20 of the solver's displacement, PREM, viscous core delta = 0.25 km, K 168, interior in
16 layers N 15, at 0.9175 cpsd (left) and M2 (right): flattened (top) and a sphere (bottom; no flattening and no
centrifugal force, Coriolis kept). Both rows over the median of the flattened map of the same frequency, so the
colours compare energies. Cyan: the traced attractor at 0.9175. From saved results:

    .venv/bin/python paper_pepi/make_figS_sphere_flat.py   -> paper_pepi/figures/figS_sphere_flat.pdf

(scripts/core_flow_attractor.py: results/core_flow_attractor_K168_S16N15.npz, _M2.npz and, with --sphere,
_sphere.npz; band 8, solid_P 0, --full-maps)
"""
import numpy as np
import matplotlib.pyplot as plt

from make_fig_rays_buoyancy import FIG, RES, W2COL, INK, LOOP

COLS = ((0.9175, 'core_flow_attractor_K168_S16N15.npz', '0.9175'),
        (1.92700, 'core_flow_attractor_M2.npz', 'M$_2$'))


def main():
    sph = np.load(RES / 'core_flow_attractor_sphere.npz')
    fig, axs = plt.subplots(2, 2, figsize=(0.62 * W2COL, 0.62 * W2COL + 0.5))
    t = np.linspace(0, np.pi / 2, 200)
    for j, (f, src, title) in enumerate(COLS):
        k = f'PREM_{f}_d0.25_K168_S16_N15'
        flat = np.load(RES / src)
        med = np.median(flat['E_' + k])
        for i, (d, nm) in enumerate(((flat, 'flattened'), (sph, 'sphere'))):
            ax = axs[i, j]
            r, th, E = d['r_' + k], d['th_' + k], d['E_' + k]
            up = th <= np.pi / 2 + 1e-9
            R, T = np.meshgrid(r / 3480e3, th[up], indexing='ij')
            pc = ax.pcolormesh(R * np.sin(T), R * np.cos(T), np.log10(E[:, up] / med), vmin=-1.5, vmax=1.5,
                               cmap='magma', shading='gouraud', rasterized=True)
            if f < 1.5:
                q = np.abs(d['loop_' + k])
                jump = np.flatnonzero(np.hypot(*np.diff(q, axis=0).T) > 0.02)
                for a, b in zip(np.r_[0, jump + 1], np.r_[jump + 1, len(q)]):
                    ax.plot(q[a:b, 0], q[a:b, 1], color=LOOP, lw=0.6)
            for rad in (1.0, 1221.5 / 3480.0):
                ax.plot(rad * np.sin(t), rad * np.cos(t), color=INK, lw=0.7)
            ax.set_aspect('equal')
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.axis('off')
            ax.set_title(f'({"abcd"[2 * i + j]}) {title}, {nm}', loc='left', fontsize=7)
    fig.subplots_adjust(bottom=0.14, wspace=0.05, hspace=0.12)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.2, 0.06, 0.6, 0.025]), orientation='horizontal')
    cb.set_label(r'$\log_{10}$ of $|\mathbf{u}|^2$ (degrees $\geq20$) over the median of the flattened core',
                 fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / 'figS_sphere_flat.pdf', bbox_inches='tight')
    fig.savefig(RES / 'figS_sphere_flat_preview.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    main()
