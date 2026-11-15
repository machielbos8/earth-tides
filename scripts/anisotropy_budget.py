# -*- coding: utf-8 -*-
#
# Transverse isotropy of PREM against the latitude dependence of the flattened Earth: amplitude of
# the M2 displacement each produces, radial, north and east, in mm (Online Resource 1).
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
# 15/10/2026 Machiel Bos
#===============================================================================
"""Anisotropy: elastic PREM at M2 with astyanax computelove (Bos & Scherneck 2013), isotropic PREM.dat
against transversely isotropic PREMtrans.dat (24-220 km), degree 2, body tide; rebuilt in a scratch
directory (astyanax untouched):
    PREM       h = 0.60614060  l = 0.16800390/2  k = 0.59876289/2
    PREMtrans  h = 0.60581666  l = 0.16795089/2  k = 0.59869052/2
Flattening: latitude-dependent parameters of the semidiurnal band, elastic hydrostatic PREM
(results/degree2_PREMs.txt, m = 2): h(2) = -0.000488, l(2) = +0.000092, l(1) = +0.000657.
Amplitudes from the M2 line of the Tamura catalogue with the formulas of etide.displacement.
Usage: python scripts/anisotropy_budget.py
"""
import numpy as np
from earthtides.etide import Tamura, W_functions, CV, RE, GE

DH = 0.60581666 - 0.60614060
DL = (0.16795089 - 0.16800390) / 2
DK = (0.59869052 - 0.59876289) / 2
H2, L2, L1 = -0.000488, 0.000092, 0.000657


def main():
    cat = Tamura()
    s = np.flatnonzero((cat.n == 2) & (cat.m == 2))
    A = cat.A[s[np.argmax(np.abs(cat.A[s]))]]                     # M2
    print(f'dh = {DH:+.2e}, dl = {DL:+.2e}, dk = {DK:+.2e}')
    print('lat   anisotropy: up north east (mm)   flattening: up north east (mm)')
    for lat in (0, 20, 45, 60):
        phi = np.radians(lat)
        sp, cp = np.sin(phi), np.cos(phi)
        W, dW = (A * x for x in W_functions(2, 2, phi))
        P2 = 1.5 * sp**2 - 0.5
        Wn = A * CV * RE / (GE / RE**2) / 3
        ani = (DH * W, DL * dW, 2 * DL * W / cp)
        fl = (H2 * P2 * W, L2 * P2 * dW + L1 * (-6 * sp * cp) * Wn,
              abs(2j * L2 * P2 * W / cp + L1 * 6j * cp * sp**2 * Wn))
        print(f'{lat:3d}   ' + ' '.join(f'{abs(v) * 1e3:6.3f}' for v in ani) + '      '
              + ' '.join(f'{abs(v) * 1e3:6.3f}' for v in fl))


if __name__ == '__main__':
    main()
