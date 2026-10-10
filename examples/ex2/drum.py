# -*- coding: utf-8 -*-
#
# Example 2: the loaded elliptical drum - Chebyshev collocation and Galerkin
# projection together.
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
# 7/10/2026 Machiel Bos
#===============================================================================
"""Example 2: the loaded elliptical drum - Chebyshev collocation and Galerkin projection together.

A membrane is stretched over an elliptical frame x^2/a^2 + y^2/b^2 = 1 (a = 1, b = 1 - f, f the
flattening), fixed at the rim, and loaded by f(x, y) (paper, Sect. 3.5):

    -lap u = load   inside the ellipse,      u = 0 on the rim.

Level-curve coordinates (as the level surfaces of the Earth): r = s R(theta), R the radius of the rim,
so the rim is s = 1 and every curve s = const is a similar ellipse. With q = R'/R,

    R^2 lap u = (1 + q^2) u_ss + (1 + q^2 - q') u_s / s + u_tt / s^2 - 2 q u_st / s .

The flattening enters only through q(theta) ~ f sin 2 theta, which couples cos 2k theta to its
neighbours, as mu_2 P2 couples degrees on the Earth.
Trial function u = sum_k U_k(s) cos 2k theta (k = 0 .. K-1; the loads are even in x and y).
  - Chebyshev collocation in s (Gauss-Lobatto nodes on [0, 1]): the equation holds at the interior
    nodes; U_k(1) = 0 at the rim; at the centre U_0'(0) = 0 and U_k(0) = 0 for k > 0 (regularity).
  - Galerkin projection in theta: the residual has no cos 2j theta component for the kept j
    (Fourier coefficients by the trapezoidal rule, exact for these smooth periodic functions).
Exact solutions for comparison (x^2/a^2 + y^2/b^2 = s^2 on the level curves):
  - uniform load 1:            u = (1 - s^2) / (2 A),  A = 1/a^2 + 1/b^2   (k = 0 only!)
  - quadrupole load x^2 - y^2: u = (1 - s^2)(alpha + beta x^2 + gamma y^2), alpha, beta, gamma from
                               a 3 x 3 system; needs all harmonics k (exponential convergence).
First-order perturbation theory in f (u0 + f du/df at f = 0) is shown for comparison.
Run:  python drum.py   ->  drum.out
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def cheb(N):
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1] (descending) and differentiation matrix (Trefethen)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(N + 1))
    return x, D - np.diag(D.sum(1))


def exact(load, a, b, x, y):
    A = 1 / a**2 + 1 / b**2
    g = 1 - x**2 / a**2 - y**2 / b**2
    if load == 'uniform':
        return g / (2 * A)
    # lap[g (alpha + beta x^2 + gamma y^2)] = -(x^2 - y^2): constant, x^2 and y^2 coefficients
    M = np.array([[-2 * A, 2, 2],
                  [0, -2 * A - 10 / a**2, -2 / a**2],
                  [0, -2 / b**2, -2 * A - 10 / b**2]])
    al, be, ga = np.linalg.solve(M, [0, -1, 1])
    return g * (al + be * x**2 + ga * y**2)


def solve(load, f, K, N, M=256):
    """Galerkin (theta) - Chebyshev collocation (s) solution; returns U[k, node] and the s nodes."""
    a, b = 1.0, 1.0 - f
    xc, D = cheb(N)
    s = (xc + 1) / 2                                      # s in [0, 1], s[0] = 1 (rim), s[N] = 0
    D1, D2 = 2 * D, 4 * D @ D
    th = 2 * np.pi * np.arange(M) / M
    R = 1 / np.sqrt(np.cos(th)**2 / a**2 + np.sin(th)**2 / b**2)
    c2, sn2 = np.cos(2 * th), np.sin(2 * th)
    e = 1 / b**2 - 1 / a**2
    dR = -(e / 2) * R**3 * sn2                             # R' for R^-2 = cos^2/a^2 + sin^2/b^2
    q = dR / R                                             # q = -(e/2) R^2 sin 2t
    dq = -(e / 2) * (2 * R * dR * sn2 + 2 * R**2 * c2)     # q'
    proj = lambda F, j: (1 if j == 0 else 2) * np.mean(F * np.cos(2 * j * th))   # Fourier coefficient
    if load == 'uniform':
        rhs_t = lambda S: -np.outer(np.ones_like(S), R**2)
    else:                                                  # x^2 - y^2 = s^2 R^2 cos 2t
        rhs_t = lambda S: -np.outer(S**2, R**4 * c2)
    nn = N + 1
    H = np.zeros((K * nn, K * nn))
    F = np.zeros(K * nn)
    inner = np.arange(1, N)                                # interior nodes
    Rhs = rhs_t(s)
    for j in range(K):
        rows = j * nn + np.arange(nn)
        for k in range(K):
            ck, sk = np.cos(2 * k * th), np.sin(2 * k * th)
            P1 = proj((1 + q**2) * ck, j)
            P2 = proj((1 + q**2 - dq) * ck + 4 * k * q * sk, j)
            P3 = proj(-4 * k**2 * ck, j)
            with np.errstate(divide='ignore', invalid='ignore'):     # s = 0 row: not used
                blk = P1 * D2 + P2 * D1 / s[:, None] + P3 * np.diag(1 / s**2)
            H[np.ix_(rows[inner], k * nn + np.arange(nn))] = blk[inner]
        F[rows[inner]] = [proj(Rhs[i], j) for i in inner]
        H[rows[0], j * nn + 0] = 1.0                       # rim: U_j(1) = 0
        if j == 0:
            H[rows[N], j * nn + np.arange(nn)] = D1[N]     # centre: U_0'(0) = 0
        else:
            H[rows[N], j * nn + N] = 1.0                   # centre: U_j(0) = 0
    U = np.linalg.solve(H, F).reshape(K, nn)
    return U, s


def max_error(load, f, U, s, nth=181):
    a, b = 1.0, 1.0 - f
    th = np.linspace(0, np.pi / 2, nth)
    R = 1 / np.sqrt(np.cos(th)**2 / a**2 + np.sin(th)**2 / b**2)
    u = sum(np.outer(U[k], np.cos(2 * k * th)) for k in range(len(U)))
    X, Y = np.outer(s, R * np.cos(th)), np.outer(s, R * np.sin(th))
    ue = exact(load, a, b, X, Y)
    return np.abs(u - ue).max() / np.abs(ue).max()


def first_order_error(load, f, nth=181, ns=41, h=1e-6):
    """Error of u(0) + f du/df|_0 against the exact solution (perturbation theory in f)."""
    th = np.linspace(0, np.pi / 2, nth)
    s = np.linspace(0, 1, ns)
    R = 1 / np.sqrt(np.cos(th)**2 + np.sin(th)**2 / (1 - f)**2)
    X, Y = np.outer(s, R * np.cos(th)), np.outer(s, R * np.sin(th))
    u0 = exact(load, 1.0, 1.0, X, Y)
    du = (exact(load, 1.0, 1.0 - h, X, Y) - exact(load, 1.0, 1.0 + h, X, Y)) / (2 * h)
    ue = exact(load, 1.0, 1.0 - f, X, Y)
    return np.abs(u0 + f * du - ue).max() / np.abs(ue).max()


if __name__ == '__main__':
    lines = []
    out = lambda t='': (print(t), lines.append(t))
    out('Loaded elliptical drum: -lap u = load, u = 0 on x^2 + y^2/(1-f)^2 = 1; Chebyshev collocation in s,')
    out('Galerkin projection on cos 2k theta; max relative error against the exact solution')
    for load in ('uniform', 'quadrupole'):
        out()
        out(f'{load} load:')
        out(f"{'f':>8s} {'K=1':>9s} {'K=2':>9s} {'K=3':>9s} {'K=4':>9s} {'K=6':>9s} {'K=8':>9s}   first-order theory")
        for f in (1 / 300, 0.1, 0.3):
            errs = [max_error(load, f, *solve(load, f, K, 24)) for K in (1, 2, 3, 4, 6, 8)]
            out(f'{f:8.4f} ' + ' '.join(f'{v:9.1e}' for v in errs) + f'   {first_order_error(load, f):9.1e}')
    U, s = solve('quadrupole', 0.1, 4, 24)
    out()
    out('quadrupole load, f = 0.1, K = 4: amplitude of each harmonic (max over s)')
    out('   ' + '  '.join(f'cos {2 * k}t: {np.abs(U[k]).max():.2e}' for k in range(4)))
    (HERE / 'drum.out').write_text('\n'.join(lines) + '\n')
