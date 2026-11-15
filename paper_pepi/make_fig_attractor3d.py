# -*- coding: utf-8 -*-
#
# Figure of paper 1 (PEPI, v2), Supplementary Material: the attractors of Fig. characteristics in three
# dimensions.
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
# 01/11/2026 Machiel Bos
#===============================================================================
"""The attractors of Fig. characteristics (b, d) rotated about the rotation axis. The core is axisymmetric,
so the ray geometry is the same in every meridional plane, and each straight segment of the attractor sweeps
a cone (a double cone if it crosses the axis). The liquid core is cut open over a quarter; on the two cut
faces the attractor is the 2-D one of the main figure. Blue: orbits of fluid parcels on a few cones. In a
thin sheet the flow varies much faster across the sheet than along it, so locally it is a plane inertial wave
with its wave vector normal to the sheet: the parcels move on circles in the plane of the sheet, clockwise
seen from above the sheet (anticyclonic), with phase omega t + m lambda, so the pattern travels around the
axis with the tide. The radius of the circles is arbitrary. Run from the project root:

    .venv/bin/python paper_pepi/make_fig_attractor3d.py   -> paper_pepi/figures/figS_attractor3d.pdf
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig_characteristics import trace, ETA, LINES, FIG, W2COL, C2, INK, MUTED

DARK = '#a8401a'
C1 = '#2a78d6'
ORBITS = {1: (0, 2, 5), 2: (1, 4, 9, 12)}         # m -> segments that get orbits


def attractor(f):
    """One period of the limit cycle: the last segments of a long ray, cut where the end point first comes
    back. A true attractor closes to round-off; a near-return of a ray that keeps filling the shell (M2 in a
    neutral core: no attractor, 2026-11-08) is refused."""
    tail = trace(np.arcsin(f / 2), ETA, 20000, (0.62, 0.15))[-400:]
    for k in range(2, 60):
        if np.linalg.norm(tail[-1] - tail[-1 - k]) < 1e-6:
            return tail[-1 - k:]
    raise RuntimeError(f'no period found at {f} cpsd')


def orbits(ax, a, b, m):
    """Parcel orbits on the cone swept by segment a-b, at two points along it and five longitudes."""
    d = np.array([np.sign(a[0] + b[0]) * (b - a)[0], (b - a)[1]])   # along the segment, (|x|, z) half plane
    d /= np.linalg.norm(d)
    r, psi = 0.05, np.linspace(0, 2 * np.pi, 60)
    for s0 in (0.3, 0.7):
        p = a + s0 * (b - a)
        if abs(p[0]) < 0.08:
            continue
        for lam in np.linspace(0.15, 1.35, 5) * np.pi:
            er, el = np.array([np.cos(lam), np.sin(lam), 0.0]), np.array([-np.sin(lam), np.cos(lam), 0.0])
            c = abs(p[0]) * er + np.array([0, 0, p[1]])
            e1 = d[0] * er + np.array([0, 0, d[1]])                     # in the sheet, along the segment
            kn = np.cross(e1, el)
            kn *= np.sign(kn[2]) if kn[2] != 0 else 1.0                 # sheet normal, upward
            e2 = np.cross(kn, e1)                                       # e1, e2, kn right-handed
            orb = c[:, None] + r * (np.outer(e1, np.sin(psi)) + np.outer(e2, np.cos(psi)))
            ax.plot(*orb, color=C1, lw=0.5)
            ps = m * lam + np.linspace(0, 0.9, 12)                      # the parcel now, and where it goes
            arc = c[:, None] + r * (np.outer(e1, np.sin(ps)) + np.outer(e2, np.cos(ps)))
            ax.plot(*arc, color=C1, lw=1.5)
            ax.scatter(*arc[:, 0], color=C1, s=4, depthshade=False)


def main():
    fig = plt.figure(figsize=(0.55 * W2COL, 3.7))
    phi = np.linspace(0.0, 1.5 * np.pi, 120)           # three quarters: cut away to look inside
    u, v = np.meshgrid(np.linspace(0, 1.5 * np.pi, 40), np.linspace(0, np.pi, 30))
    tt = np.linspace(0, np.pi, 100)
    for j, ((name, f), lab, m) in enumerate(zip(LINES[:1], 'a', (1,))):   # M2 has no attractor
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        A = attractor(f)
        t = np.linspace(0, 1, 40)
        for i in range(len(A) - 1):
            x = A[i, 0] + t * (A[i + 1, 0] - A[i, 0])
            z = A[i, 1] + t * (A[i + 1, 1] - A[i, 1])
            R, PH = np.meshgrid(np.abs(x), phi)
            Z = np.meshgrid(z, phi)[0]
            ax.plot_surface(R * np.cos(PH), R * np.sin(PH), Z, color=C2, alpha=0.16, linewidth=0,
                            antialiased=True, shade=True, rasterized=True)
            for p0 in (phi[0], phi[-1]):                 # the 2-D attractor on the two cut faces
                ax.plot(np.abs(x) * np.cos(p0), np.abs(x) * np.sin(p0), z, color=DARK, lw=1.2)
            if i in ORBITS[m]:
                orbits(ax, A[i], A[i + 1], m)
        ax.plot_wireframe(np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v), color=MUTED,
                          lw=0.25, rstride=3, cstride=4, rasterized=True)
        ax.plot_surface(ETA * np.sin(v) * np.cos(u), ETA * np.sin(v) * np.sin(u), ETA * np.cos(v),
                        color='#bdbcb6', alpha=0.9, linewidth=0, rasterized=True)
        for p0 in (0.0, 1.5 * np.pi):                   # outline of the two cut faces
            for r in (1.0, ETA):
                ax.plot(r * np.sin(tt) * np.cos(p0), r * np.sin(tt) * np.sin(p0), r * np.cos(tt), color=INK, lw=0.7)
        ax.plot([0, 0], [0, 0], [-1.25, 1.25], color=INK, ls='--', lw=0.7)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=22, azim=-35)
        ax.set_axis_off()
        ax.set_title(f'{name}, $m={m}$: {len(A) - 1} cones', fontsize=8)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'figS_attractor3d.pdf', dpi=300)


if __name__ == '__main__':
    main()
