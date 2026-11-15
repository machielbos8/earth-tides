# -*- coding: utf-8 -*-
#
# Figure of paper 1 (PEPI, v2): characteristics of inertial waves in the liquid core, with and without
# the inner core.
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
"""Characteristics of the inviscid Poincare equation in the meridional plane of the core. At frequency
omega (rotating frame) inside |omega| < 2 Omega the equation is hyperbolic and its characteristics are
straight lines at the angle arcsin(omega / 2 Omega) to the rotation axis; at a wall a ray changes family
(keeps that angle). In a full sphere every ray stays spread over the fluid; in a shell (inner core, radius
ratio 1221.5 / 3480 = 0.351) rays are focused onto a limit cycle, the attractor (Rieutord et al. 2001), in
frequency windows that alternate with non-focusing frequencies (O1 at 0.927 cpsd falls between two).
The flattening (1/400) does not change the picture. Run from the project root:

    .venv/bin/python paper_pepi/make_fig_characteristics.py   -> paper_pepi/figures/fig_characteristics.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper_pepi' / 'figures'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#8a8984'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W2COL = 7.2
ETA = 1221.5 / 3480.0          # inner-core / core-mantle-boundary radius, PREM
# cycles per sidereal day, rotating frame: diurnal (0.907, between Q1 and O1; O1 at 0.927 falls between two attractor windows) and M2
LINES = (('0.907 cpsd', 0.907), ('M$_2$, 1.927 cpsd', 1.92700))


def _hit(p, d, R, eps=1e-12):
    """Smallest t > eps with |p + t d| = R, or inf."""
    b, c = p @ d, p @ p - R * R
    disc = b * b - c
    if disc < 0:
        return np.inf
    q = np.sqrt(disc)
    ts = [t for t in (-b - q, -b + q) if t > eps]
    return min(ts) if ts else np.inf


def trace(beta, eta, n, p0):
    """n segments of a ray in the meridional annulus eta < |p| < 1 (eta = 0: full disc); characteristics at
    the angle beta to the z axis. Returns the reflection points."""
    sb, cb = np.sin(beta), np.cos(beta)
    p = np.array(p0, float)
    d = np.array([sb, -cb])
    pts = [p.copy()]
    for _ in range(n):
        t_out = _hit(p, d, 1.0)
        t_in = _hit(p, d, eta) if eta > 0 else np.inf
        t = min(t_out, t_in)
        p = p + t * d
        nin = -p / np.linalg.norm(p) if t_out <= t_in else p / np.linalg.norm(p)   # into the fluid
        # change family: flip one component, keeping the angle to the axis; take the inward one
        cands = [np.array([-d[0], d[1]]), np.array([d[0], -d[1]])]
        d = max(cands, key=lambda c: c @ nin)
        pts.append(p.copy())
    return np.array(pts)


N_SAMPLE = 300


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(W2COL * 0.62, W2COL * 0.62))
    th = np.linspace(0, 2 * np.pi, 400)
    n_all = 20000
    p0 = (0.62, 0.15)                     # a point in the fluid
    for i, (name, f) in enumerate(LINES):
        beta = np.arcsin(f / 2)
        for j, eta in enumerate((0.0, ETA)):
            ax = axs[i, j]
            # with the inner core the ray starts where beams are launched: its critical latitude, where a
            # characteristic grazes the inner core (latitude beta); without it, a test ray at p0
            start = (eta * (1 + 1e-9)) * np.array([np.cos(beta), np.sin(beta)]) if eta else np.array(p0)
            P = trace(beta, eta, n_all, start)
            seg = np.stack([P[:-1], P[1:]], axis=1)
            # N_SAMPLE segments drawn at random from all n_all reflections (fixed seed): an even sample of
            # where the energy goes; a regular stride would alias with the period of the reflections
            pick = np.sort(np.random.default_rng(1).choice(len(seg), N_SAMPLE, replace=False))
            ax.add_collection(LineCollection(seg[pick], colors=C2 if eta else C1, linewidths=0.35, alpha=0.6))
            ax.plot(np.cos(th), np.sin(th), color=INK, lw=0.8)
            if eta:
                ax.fill(eta * np.cos(th), eta * np.sin(th), color='#d9d8d3', lw=0)
                ax.plot(eta * np.cos(th), eta * np.sin(th), color=INK, lw=0.8)
            ax.plot([0, 0], [-1.12, 1.12], color=INK, lw=0.5, ls=(0, (4, 3)))
            ax.plot(*start, 'o', color=INK, ms=3.5, zorder=5)
            ax.set_aspect('equal')
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.14, 1.14)
            ax.axis('off')
            ax.set_title(f'{"(" + "abcd"[2 * i + j] + ")"} {name}, '
                         f'{"no inner core" if not eta else "with inner core"}', fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C1, lw=1.0), Line2D([], [], color=C2, lw=1.0)]
    fig.legend(handles, [f'{N_SAMPLE} of {n_all} reflections at random: no inner core', 'with inner core'],
               loc='lower center', ncol=2, fontsize=7, handlelength=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=0.4, h_pad=0.6, rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / 'fig_characteristics.pdf')


if __name__ == '__main__':
    main()
