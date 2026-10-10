# -*- coding: utf-8 -*-
#
# Example 1: the pumped swing - residuals and projection in one variable.
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
# 6/10/2026 Machiel Bos
#===============================================================================
"""Example 1: the pumped swing - residuals and projection in one variable.

A swing is pushed once per period while its rider crouches at both ends and
stands up at the bottom (paper, Sect. 3.4):

    x'' + w0^2 (1 + eps cos 2t) x = cos t,     w0^2 = 2, eps = -0.1.

Trial function x = a1 cos t + a3 cos 3t + ... (K odd harmonics). The residual
E(t) = left-hand side - right-hand side is required to contain none of the kept
harmonics (Galerkin projection), which gives K linear equations for K unknowns.
The result is compared with the periodic solution from numerical integration
and with first-order perturbation theory.

Run:  python swing.py        ->  swing.out  (and residual.png if matplotlib is installed)
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
W2, EPS = 2.0, -0.1
T = np.linspace(0, 2 * np.pi, 400, endpoint=False)      # exact quadrature for these trig. polynomials


def harmonics(K):
    return 2 * np.arange(K) + 1                             # 1, 3, 5, ...


def contribution(n, t):
    """What the basis function cos(nt) contributes to the left-hand side."""
    return (W2 * (1 + EPS * np.cos(2 * t)) - n**2) * np.cos(n * t)


def galerkin(K):
    """Amplitudes a_n from the projected residual: H a = f."""
    n = harmonics(K)
    project = lambda E, k: 2 * np.mean(E * np.cos(k * T))  # Fourier coefficient of cos(kt)
    H = np.array([[project(contribution(nj, T), k) for nj in n] for k in n])
    f = np.array([project(np.cos(T), k) for k in n])
    return np.linalg.solve(H, f), H


def residual(a, t):
    n = harmonics(len(a))
    return sum(ai * contribution(ni, t) for ai, ni in zip(a, n)) - np.cos(t)


def periodic_solution():
    """x(0) of the 2 pi-periodic solution by numerical integration (independent check)."""
    rhs = lambda t, y: [y[1], np.cos(t) - W2 * (1 + EPS * np.cos(2 * t)) * y[0]]
    flow = lambda y0: solve_ivp(rhs, [0, 2 * np.pi], y0, rtol=1e-12, atol=1e-14).y[:, -1]
    z = flow([0.0, 0.0])
    M = np.column_stack([flow([1.0, 0.0]) - z, flow([0.0, 1.0]) - z])
    return np.linalg.solve(np.eye(2) - M, z)[0]


if __name__ == '__main__':
    lines = []
    out = lambda s='': (print(s), lines.append(s))
    ex, _ = galerkin(30)                                  # converged to round-off
    a2, H2 = galerkin(2)
    out('Pumped swing: x\'\' + w0^2 (1 + eps cos 2t) x = cos t,  w0^2 = 2, eps = -0.1')
    out()
    out('K = 2 system (rows: projections on cos t, cos 3t; columns: a1, a3):')
    for row in H2:
        out('   ' + '  '.join(f'{v:+9.5f}' for v in row))
    out('   the off-diagonal entry is the coupling coefficient eps w0^2 / 2 = %+.5f' % (EPS * W2 / 2))
    out()
    out(f'x(0) = sum a_n: numerical integration {periodic_solution():.12f}, Galerkin K=30 {ex.sum():.12f}')
    out(f'converged amplitudes: a1 = {ex[0]:.10f}, a3 = {ex[1]:.10f}, a5 = {ex[2]:.3e}')
    out(f'without pumping (eps = 0): a1 = {1 / (W2 - 1):.10f}')
    out()
    out(' K   a1               |error a1|   max |E(t)|')
    for K in (1, 2, 3, 4):
        a, _ = galerkin(K)
        out(f' {K}   {a[0]:.10f}   {abs(a[0] - ex[0]):9.1e}   {np.abs(residual(a, T)).max():9.1e}')
    a1p = 1 / (W2 - 1) * (1 - EPS * W2 / 2 / (W2 - 1))
    out(f'first-order perturbation theory: a1 = {a1p:.10f}, |error| = {abs(a1p - ex[0]):.1e}')
    (HERE / 'swing.out').write_text('\n'.join(lines) + '\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit
    t = np.linspace(0, 2 * np.pi, 600)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for K, ls in ((1, '-'), (2, '--'), (3, ':')):
        a, _ = galerkin(K)
        E = residual(a, t)
        ax.plot(t / np.pi, E / np.abs(E).max(), ls, color='k',
                label=f'K = {K}: max |E| = {np.abs(E).max():.1e}')
    ax.set_xlabel('t / π')
    ax.set_ylabel('E(t) / max |E|')
    ax.set_title('Residual: K = 1 leaves cos 3t, K = 2 leaves cos 5t, ...', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / 'residual.png', dpi=150)
