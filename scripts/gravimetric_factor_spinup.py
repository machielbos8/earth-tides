# -*- coding: utf-8 -*-
#
# Diurnal gravimetric factors of a hydrostatic PREM whose free core nutation is at the observed
# frequency, obtained consistently by spinning the model slightly faster.
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
# 9/10/2026 Machiel Bos
#===============================================================================
"""Gravimetric factors and load resonance with the FCN at 431.4 sidereal days, consistently.

The FCN offset sigma_FCN - 1 grows with the dynamical ellipticity of the core. Imposing a larger
flattening on the map breaks the hydrostatic reference state (fcn_period.py). A consistent way to
raise it is to spin the hydrostatic model faster, Omega' = Omega (1 + x): all level surfaces
flatten as Omega'^2 and the reference state stays hydrostatic. Frequencies are then in cycles per
sidereal day OF THAT EARTH (O1 = 0.92700, K1 = 1), which is the scale on which the resonance acts.
x = 0.0327 puts the FCN of anelastic PREM at 1.0023181 cpsd (TN36 eq. 6.10). Side effects: the
surface flattening becomes 1/281 instead of 1/300, which changes the ellipticity part of delta0 by
a few 1e-5, nearly equally for O1 and K1.
Computed: delta(O1), delta(K1) at station latitudes (anelastic, no ocean) and the K1 load
resonance dh', dk' (rotating minus spherical), for x = 0 and x = 0.0327; then
rho_pred = (3 rho_w/5 rho_bar) Re[A21 (dh' - 1.5 dk')] / delta(K1) for both admittance signs.
K1 sits on the tilt-over frequency; it is evaluated as the mean of sigma = 1 -+ 2e-4.
Usage: python scripts/gravimetric_factor_spinup.py -> results/gravimetric_factor_spinup.txt
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.gravimetry import surface_gravity
from earthtides import etide

STATIONS = {'Membach': 50.609, 'Strasbourg': 48.622, 'Wettzell': 49.146, 'Vienna': 48.248,
            'Medicina': 44.522, 'Brussels': 50.799, 'Metsahovi': 60.217, 'Bad Homburg': 50.229,
            'Moxa': 50.645, 'Pecny': 49.920, 'Conrad': 47.928}   # Ducarme et al. 2009 Table 1; 2014
A21_K1 = -0.2120 - 0.2276j            # FES2022b (results/otl_fcn_resonance.txt)
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def kw(md, geo, Om, sigma, forcing='tide'):
    w = sigma * Om
    S = TidalSolver(md.anelastic(w, dispersion='cot'), geo, m=1, K=4, N=24, omega_f=0.0,
                    coriolis=Om, omega_t=w, omega_rot=Om, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle', forcing=forcing)
    S.solve()
    return S


def delta(md, geo, Om, sigma, colat):
    th, dg, dgT = surface_gravity(kw(md, geo, Om, sigma))
    f = np.sqrt(3) / (2 * np.sqrt(2)) * (7 * np.cos(th)**2 - 3)
    d0, dp = np.linalg.lstsq(np.column_stack([dgT, dgT * f]), dg, rcond=None)[0]
    return np.array([d0 + dp * np.sqrt(3) / (2 * np.sqrt(2)) * (7 * np.cos(t)**2 - 3) for t in colat]), d0, dp


def load_res(md, geo, Om, sigma):
    S = kw(md, geo, Om, sigma, 'load')
    R = TidalSolver(md.anelastic(sigma * Om, dispersion='cot'), m=0, n_force=2, degrees=([2], []),
                    omega_f=sigma * Om, dyn_fluid=True, forcing='load')
    R.solve()
    a, b = S.load_love(), R.load_love()
    return a['h'][2] - b['h'][2], a['k'][2] - b['k'][2]


def main():
    md = EarthModel(str(model_path('PREM')))
    names = list(STATIONS)
    colat = [np.pi / 2 - etide.geocentric(STATIONS[n])[0] for n in names]
    fac = 0.6 * 1025.0 / (3 * md.M / (4 * np.pi * md.a**3))
    out('Diurnal gravimetric factors (ICET), anelastic hydrostatic PREM; K1 = mean of sigma 1 -+ 2e-4')
    res = {}
    for x in (0.0, 0.0327):
        Om = OMEGA_EARTH * (1 + x)
        geo = ClairautGeometry(md, omega=Om)
        dO, dO0, dOp = delta(md, geo, Om, 0.92700, colat)
        k_ = [delta(md, geo, Om, s, colat) for s in (1 - 2e-4, 1 + 2e-4)]
        dK = 0.5 * (k_[0][0] + k_[1][0])
        dK0, dKp = 0.5 * (k_[0][1] + k_[1][1]), 0.5 * (k_[0][2] + k_[1][2])
        lr = [load_res(md, geo, Om, s) for s in (1 - 2e-4, 1 + 2e-4)]
        dh = 0.5 * (lr[0][0] + lr[1][0])
        dk = 0.5 * (lr[0][1] + lr[1][1])
        ot = fac * A21_K1 * (dh - 1.5 * dk)
        res[x] = (dO, dK, ot)
        out()
        out(f'x = {x}: surface flattening 1/{1/geo.flattening_surface:.1f}')
        out(f'  O1: delta0 {dO0.real:.5f}{dO0.imag:+.5f}i  delta+ {dOp.real:+.5f}')
        out(f'  K1: delta0 {dK0.real:.5f}{dK0.imag:+.5f}i  delta+ {dKp.real:+.5f}')
        out(f"  K1 load resonance dh' {dh.real:+.4f}  dk' {dk.real:+.4f}  ocean term (our sign) "
            f"{ot.real:+.5f}{ot.imag:+.5f}i")
        for j, n in enumerate(names):
            out(f'  {n:11s} O1 {dO[j].real:.5f}  K1 {dK[j].real:.5f}  K1/O1 {dK[j].real/dO[j].real:.6f}'
                f'  rho_pred ours {100*ot.real/dK[j].real:+.3f} %  IERS sign {-100*ot.real/dK[j].real:+.3f} %')
    out()
    r0 = res[0.0][1][0].real / res[0.0][0][0].real
    r1 = res[0.0327][1][0].real / res[0.0327][0][0].real
    out(f'Membach K1/O1: FCN at 460 sd {r0:.6f}, at 431 sd {r1:.6f}: change {100*(r1/r0-1):+.3f} %')
    Path('results/gravimetric_factor_spinup.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
