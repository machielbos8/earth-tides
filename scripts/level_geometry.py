# -*- coding: utf-8 -*-
#
# Geometry whose coordinate surfaces are level surfaces of the reference potential the solver uses.
#
# This file is part of earth-tides 1.1.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 14/11/2026 Machiel Bos
#===============================================================================
"""The first-order (Clairaut) map and the exact ellipses of the same flattening are level surfaces of the
solver's reference potential (refpot.ReferencePotential: Poisson solve for the density on the mapped grid,
degrees 0 - 6, plus the exact centrifugal potential) only to metres (am_wall_level.py: CMB 7.2 and 2.2 m).
LevelGeometry starts from the exact ellipses and adds R += sum_n d_n(s) P_n(cos theta), n = 2, 4, 6, with
d_n a Chebyshev interpolant per layer (M + 1 Lobatto nodes, the end values shared with the neighbouring layer:
the map stays continuous). Iteration: reference potential of the current figure -> height of every
coordinate surface above the level surface through its mean, h = (Phi0 - mean) / (dPhi0/dr), at the nodes ->
d_n -= relax h_n, until max |h| < tol or itmax (h halves per step: moving the surfaces moves the potential;
relax 1.9 was slower, 0.85 per step). The density stays rho0(s): the figure is then in hydrostatic equilibrium with
its own potential in the degrees kept (what remains, degree 8 and above, is printed).
build(md, omega, M=10, tol=1e-6 m) -> geometry, list of (iteration, max |h| in m, max of the part not in 2-6)."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np
from numpy.polynomial import chebyshev as Ch, legendre as L

_argv, sys.argv = sys.argv, sys.argv[:1]
from exact_levels_share_test import ExactLevelsGeometry
sys.argv = _argv
from earthtides.geometry import mapping_from_R
from earthtides.refpot import ReferencePotential
from earthtides.basis import cheb_basis, ybar

DEG = (2, 4, 6)


class LevelGeometry(ExactLevelsGeometry):
    """Exact ellipses plus sum_n d_n(s) P_n(cos theta); d[i][n]: Chebyshev coefficients of d_n in layer i."""

    def __init__(self, md, M=10, **kw):
        super().__init__(md, **kw)
        self.M = M
        self.d = [{n: np.zeros(M + 1) for n in DEG} for _ in range(md.n_layers)]

    def map_derivs(self, i, s, theta):
        out = list(super().map_derivs(i, s, theta))
        md = self.model
        sc = 2.0 / (md.r1[i] - md.r0[i])
        x = sc * (np.asarray(s, float) - md.r0[i]) - 1
        c, sn = np.cos(theta), np.sin(theta)
        for n in DEG:
            a = self.d[i][n]
            d0, d1, d2 = Ch.chebval(x, a), Ch.chebval(x, Ch.chebder(a)) * sc, Ch.chebval(x, Ch.chebder(a, 2)) * sc ** 2
            Pn = L.Legendre.basis(n)
            P, P1, P2 = Pn(c), Pn.deriv(1)(c), Pn.deriv(2)(c)
            Pt, Ptt = -P1 * sn, P2 * sn * sn - P1 * c
            for k, v in enumerate((d0 * P, d1 * P, d2 * P, d0 * Pt, d1 * Pt, d0 * Ptt)):
                out[k] = out[k] + v
        return tuple(out)


def heights(g, rp, n_theta=40):
    """h_n (m) of every Lobatto node of every layer: Legendre coefficients 2, 4, 6 of the height of the coordinate
    surface above the level surface through its mean; and max |h|, max |h - its degrees 2 - 6|."""
    md = g.model
    xg, wg = L.leggauss(n_theta)
    th = np.arccos(xg)
    xk = np.cos(np.pi * np.arange(g.M + 1) / g.M)[::-1]              # -1 .. 1
    H = np.zeros((md.n_layers, g.M + 1, len(DEG)))
    hmax = rest = 0.0
    for i in range(md.n_layers):
        s = md.r0[i] + (xk + 1) * (md.r1[i] - md.r0[i]) / 2
        for k, sk in enumerate(s):
            if sk < 1.0:                                              # the centre
                continue
            s2 = np.array([[sk]])
            mp = mapping_from_R(*g.map_derivs(i, s2, th))
            T = cheb_basis(np.array([xk[k]]), rp.N)[0]
            phi = sum((rp.coef[i * rp.nb + j * rp.N: i * rp.nb + (j + 1) * rp.N] @ T)[0] * ybar(n, 0, th, 0)[0]
                      for j, n in enumerate(rp.deg))
            r = mp['R'][0]
            phi = phi - rp.Om ** 2 * r ** 2 * np.sin(th) ** 2 / 2
            gr = rp.phys_derivs(i, np.array([sk]), th, mp)['r'][0]
            h = (phi - np.sum(wg * phi) / 2) / gr
            cn = [(2 * n + 1) / 2 * np.sum(wg * h * L.Legendre.basis(n)(xg)) for n in DEG]
            H[i, k] = cn
            hmax = max(hmax, np.max(np.abs(h)))
            rest = max(rest, np.max(np.abs(h - sum(c * L.Legendre.basis(n)(xg) for c, n in zip(cn, DEG)))))
    return xk, H, hmax, rest


def build(md, omega, M=10, tol=1e-6, itmax=30, relax=1.0, log=print):
    g = LevelGeometry(md, M=M)
    hist = []
    for it in range(itmax):
        rp = ReferencePotential(md, g, omega)
        xk, H, hmax, rest = heights(g, rp)
        hist.append((it, hmax, rest))
        log(f'level geometry, iteration {it}: max |h| {hmax:.2e} m, not in degrees 2-6: {rest:.2e} m')
        if hmax < tol + 2 * rest:
            break
        for i in range(md.n_layers):                                  # shared end values: continuous map
            if i + 1 < md.n_layers:
                H[i, -1] = H[i + 1, 0] = 0.5 * (H[i, -1] + H[i + 1, 0])
        for i in range(md.n_layers):
            for j, n in enumerate(DEG):
                val = Ch.chebval(xk, g.d[i][n]) - relax * H[i, :, j]
                g.d[i][n] = Ch.chebfit(xk, val, g.M)
    return g, hist
