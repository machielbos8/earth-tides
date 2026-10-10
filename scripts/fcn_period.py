# -*- coding: utf-8 -*-
#
# Free core nutation period of a model: fit k(sigma) = a + b/(sigma_FCN - sigma) near the resonance.
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
"""FCN period (sidereal days in space) of hydrostatic PREM, elastic or anelastic.

A non-hydrostatic core flattening imposed on the level-surface map (eps + d_eps in the core) was
tried on 2026-10-09 and rejected: the equations assume density surfaces = level surfaces
(grad rho0 || grad Phi0), so the rigid tilt is no longer force-free; the resonance changed sign
and k moved by 0.1 far from it. A consistent version needs a non-hydrostatic pre-stress.
Usage: python scripts/fcn_period.py [elastic|anelastic]
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver


def solver(md, geo, sigma, anelastic):
    w = sigma * OMEGA_EARTH
    m_ = md.anelastic(w, dispersion='cot') if anelastic else md
    S = TidalSolver(m_, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    return S


def fcn(md, geo, anelastic=False, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    """sigma_FCN (cpsd, real part) from k at four frequencies (linear least squares)."""
    x = np.array(sig)
    k = np.array([solver(md, geo, s, anelastic).results()['kk'] for s in sig]).real
    # k (c - x) = a (c - x) + b  ->  k x = c k + a x - d
    M = np.column_stack([k, x, -np.ones_like(x)])
    c, a, d = np.linalg.lstsq(M, k * x, rcond=None)[0]
    return c


def main():
    an = len(sys.argv) > 1 and sys.argv[1] == 'anelastic'
    md = EarthModel(str(model_path('PREM')))
    c = fcn(md, ClairautGeometry(md), an)
    print(f'sigma_FCN={c:.7f} cpsd  period={1/(c-1):.1f} sidereal days ({"anelastic" if an else "elastic"})')


if __name__ == '__main__':
    main()
