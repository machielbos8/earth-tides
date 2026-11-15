# -*- coding: utf-8 -*-
#
# b2_profile.
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
# 20/10/2026 Machiel Bos
#===============================================================================
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
sys.argv = ['x', '2', '1.9275']
exec(open('scripts/b2_coeffs.py').read().split("m, sig = int")[0])
import scipy.linalg as sla
from earthtides.model import G
md = EarthModel('data/PREM.dat'); g = ClairautGeometry(md)
i = 1; H = md.r1[i] - md.r0[i]
def N2(s):                       # buoyancy frequency squared / Omega^2 (PREM core, Adams-Williamson reference)
    rho, drho = md.rho[i](s), md.rho[i].deriv()(s)
    kap = md.lam[i](s); gg = md.gravity(i, s)
    return (-gg * (drho / rho + rho * gg / kap)) / OMEGA_EARTH**2
for m, sig in ((2, 1.9275), (1, 0.927), (2, 2.5)):
    S = TidalSolver(md, g, m=m, K=2, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True, frame='mantle')
    out = []
    for s in np.linspace(md.r0[i] + 0.02 * H, md.r1[i] - 0.02 * H, 9):
        C, flds = coeffs(S, i, [s]); C2, C1, C0 = C[2, 0], C[1, 0], C[0, 0]
        rs = np.maximum(np.abs(np.hstack([C2, C1, C0])).max(axis=1), 1e-300)
        C2, C1, C0 = (X / rs[:, None] for X in (C2, C1, C0))
        n = C2.shape[0]
        lam = sla.eigvals(np.block([[np.zeros((n, n)), np.eye(n)], [-C0, -C1]]), np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), C2]]))
        fin = lam[np.isfinite(lam) & (np.abs(lam) < 1e3 / H)]
        out.append(f'{(s - md.r0[i])/H:.2f}:{np.abs(fin.imag).max()*H/np.pi:5.1f}|N2 {N2(s):+5.2f}')
    print(f'm={m} {sig}: depth fraction above ICB : half-waves per H | N^2/Omega^2')
    print('   ' + '  '.join(out))
