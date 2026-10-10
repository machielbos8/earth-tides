# -*- coding: utf-8 -*-
#
# Fit of a computed surface displacement to the IERS 2010 model.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Fit of a computed surface displacement to the IERS 2010 model.

IERS Conventions (2010), TN36 chapter 7, eqs. (1a-c), (8), (9): parameters
h(0), h(2), l(0), l(2), l(1), h' (long period) and l' (diurnal).
Normalisation as in TN36: potential at the equatorial radius R_e divided by
g_e = GM/R_e^2. Geocentric latitude, north = -u_theta, east = u_lambda.
"""
import numpy as np

from .basis import ybar

IERS_TABLE = {0: dict(h2=-0.0006, l2=0.0002, l1=0.0, hp=0.0001),
              1: dict(h2=-0.0006, l2=0.0002, l1=0.0012, lp=-0.0002),
              2: dict(h2=-0.0006, l2=0.0002, l1=0.0024)}


def iers_fit(S, geom):
    """Least-squares fit of the surface field to TN36 eqs. (1a-c), (8), (9).

    Works for complex (anelastic) fields: the parameters come out complex.
    """
    md = S.model
    a = md.a
    e, _, _ = geom.eps(md.n_layers - 1, np.array([a]))
    norm = (a * (1 + e[0] / 3.0))**4 / a**3
    th = np.linspace(0.03, np.pi - 0.03, 600)
    ur, ut, up = (v / norm for v in S.surface_field(th))
    return iers_fit_fields(S.m, th, ur, ut, up)


def iers_fit_fields(m, th, ur, ut, up, f_geodetic=0.0):
    """The TN36 fit for given surface fields (u_r, u_theta, u_lambda amplitudes at colatitudes th,
    already normalised). f_geodetic > 0 evaluates the basis functions at the geodetic latitude
    phi + f sin 2 phi instead of the geocentric one (convention tests)."""
    north, east = -ut, up
    phi = np.pi / 2 - th
    phi = phi + f_geodetic * np.sin(2 * phi)
    sp, cp = np.sin(phi), np.cos(phi)
    P2 = 0.5 * (3 * sp**2 - 1)
    YI = {0: P2, 1: 3 * sp * cp, 2: 3 * cp**2}[m]
    C = np.median(ybar(2, m, th, 0)[0] / YI)
    # radial: C [h0 + h2 P2] YI (+ C sqrt(4pi/5) h' for m = 0)
    colsR = [C * YI, C * P2 * YI] + ([C * np.sqrt(4 * np.pi / 5) + 0 * th] if m == 0 else [])
    pr, *_ = np.linalg.lstsq(np.column_stack(colsR).astype(complex), ur.astype(complex), rcond=None)
    misR = np.abs(np.column_stack(colsR) @ pr - ur).max()
    # transverse: stack north and east
    if m == 0:
        # (1a): east = cos phi [3 l(1) sin^2 phi - sqrt(4 pi/5) l'] (zero without Coriolis)
        cn = [3 * C * sp * cp, 3 * C * P2 * sp * cp, 0 * th, 0 * th]
        ce = [0 * th, 0 * th, 3 * C * cp * sp**2, -C * np.sqrt(4 * np.pi / 5) * cp]
        names = ['l0', 'l2', 'l1', 'lp']
    elif m == 1:
        c1 = np.sqrt(24 * np.pi / 5)
        cn = [3 * C * np.cos(2 * phi), 3 * C * P2 * np.cos(2 * phi), -3 * C * sp**2, C * c1 + 0 * th]
        ce = [-3 * C * sp, -3 * C * P2 * sp, 3 * C * sp * np.cos(2 * phi), C * c1 * sp]
        names = ['l0', 'l2', 'l1', 'lp']
    else:
        cn = [-6 * C * sp * cp, -6 * C * P2 * sp * cp, -6 * C * sp * cp]
        ce = [-6 * C * cp, -6 * C * P2 * cp, -6 * C * sp**2 * cp]
        names = ['l0', 'l2', 'l1']
    At = np.vstack([np.column_stack(cn), np.column_stack(ce)])
    bt = np.concatenate([north, east])
    pt, *_ = np.linalg.lstsq(At.astype(complex), bt.astype(complex), rcond=None)
    misT = np.abs(At @ pt - bt).max()
    if not (np.iscomplexobj(ur) or np.iscomplexobj(bt)):
        pr, pt = pr.real, pt.real
    res = dict(h0=pr[0], h2=pr[1], misR=misR, misT=misT, scaleR=np.abs(ur).max(),
               scaleT=np.abs(bt).max())
    if m == 0:
        res['hp'] = pr[2]
    res.update(dict(zip(names, pt)))
    return res
