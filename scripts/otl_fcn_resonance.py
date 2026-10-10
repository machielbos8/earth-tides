"""Resonant ocean-tide-loading correction to the diurnal body-tide Love number h(0) (paper Sect. 7).

Ocean-loading displacements (BLQ, Green's functions) use load Love numbers that do not depend on
frequency. In the diurnal band the degree-2 load Love numbers resonate with the free core nutation
(Wahr & Sasao 1981), because the degree-2, order-1 part of the ocean tide exerts a torque on the
equatorial bulge and the core. The IERS Conventions (TN36 Sect. 7.1.1) put the resonant part into
the body-tide h(0) of Table 7.2 as dh(OT). Here it is computed with our own solver:

  dh(OT)(f) = (3/5) (rho_w / rho_mean) A21(f) [h'2(f) - h'2,ref(f)]

with h'2(f) the degree-2 load Love number of the rotating, flattened Earth with a fluid core
(order 1, anelastic PREM, mantle frame), h'2,ref that of the spherical non-rotating Earth (what a
Green's function uses), and A21 the admittance of the degree-2, order-1 part of the ocean tide:
the projection of the ocean tide (FES2022b, Greenwich phase lags, Doodson-Cartwright arguments) on
sin(2 phi) exp(i lambda), divided by the equilibrium tide of the same wave (Tamura amplitude, etide
normalization). A21 is computed for Q1, O1, P1, K1 and interpolated linearly in frequency between
them (held constant beyond K1).
Usage: python scripts/otl_fcn_resonance.py  ->  results/otl_fcn_resonance.txt
"""
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
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from earthtides.model import EarthModel, G
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides import etide

GRIDS = Path.home() / 'projects/otl_map/surfaceloads/FES2022b'
RHO_W = 1025.0
LINES = [('Q1', 0.893244), ('O1', 0.929536), ('P1', 0.997262), ('K1', 1.002738),
         ('psi1', 1.005476), ('phi1', 1.008214), ('J1', 1.039030), ('OO1', 1.075940)]
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def admittance_grid(amp, pha, lat, lon, amp_tamura):
    """Degree-2, order-1 admittance (ocean / equilibrium) of an ocean tide given as grids of
    amplitude and Greenwich phase lag [deg] (row = latitude [rad], column = longitude [rad]), in
    the convention of the ocean tide models and of BLQ files: height = Re[amp exp(-i pha) ...]."""
    ocean = (amp < 999.0) & (amp > 0.0)
    Z = np.where(ocean, amp * np.exp(-1j * np.radians(np.where(ocean, pha, 0.0))), 0.0)
    f, w = np.sin(2 * lat), np.cos(lat)
    num = ((Z * np.exp(-1j * lon)[None, :]).sum(1) * f * w).sum() / len(lon)
    den = (f**2 * w).sum()
    H = amp_tamura * etide.CV * etide.RE / (etide.GE / etide.RE**2)   # equilibrium tide, same pattern
    return num / den / H


def admittance(wave, amp_tamura):
    """Degree-2, order-1 admittance of the ocean tide of one wave (complex)."""
    sys.path.insert(0, 'scripts')
    from station_error_budget import read_grid
    amp, lat, lon = read_grid(GRIDS / ('amp.' + wave))
    pha, _, _ = read_grid(GRIDS / ('pha.' + wave))
    return admittance_grid(amp, pha, lat, lon, amp_tamura)


def h_load(md, geo, f_cpd):
    w = f_cpd * 2 * np.pi / 86400.0
    mda = md.anelastic(w, dispersion='cot')
    S = TidalSolver(mda, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', forcing='load')
    S.solve()
    R = TidalSolver(mda, m=0, n_force=2, degrees=([2], []), omega_f=w, dyn_fluid=True, forcing='load')
    R.solve()
    return S.load_love()['h'][2], R.load_love()['h'][2]


def main():
    md = EarthModel(str(model_path('PREM')))
    geo = ClairautGeometry(md)
    rho_mean = 3 * md.M / (4 * np.pi * md.a**3)
    fac = 0.6 * RHO_W / rho_mean
    cat = etide.Tamura()
    def tamura_amp(fc):
        k = np.argmin(np.abs(np.abs(cat.f_cpd) - fc) + 10 * ((cat.n != 2) | (cat.m != 1)))
        return cat.A[k]
    out('Resonant ocean-tide-loading correction dh(OT) to h(0), diurnal band (scripts/otl_fcn_resonance.py)')
    out(f'FES2022b ocean tides; anelastic PREM (cot dispersion); (3/5) rho_w/rho_mean = {fac:.5f}')
    out()
    out('admittance of the degree-2, order-1 ocean tide (ocean / equilibrium):')
    adm = {}
    for wave, fc in LINES[:4]:
        adm[wave] = admittance(wave.lower(), tamura_amp(fc))
        out(f'  {wave:4s} f={fc:.6f}  A21 = {adm[wave]:.4f}  (|A| {abs(adm[wave]):.4f}, '
            f'phase {np.degrees(np.angle(adm[wave])):+.1f} deg)')
    fq = np.array([fc for _, fc in LINES[:4]])
    aq = np.array([adm[w] for w, _ in LINES[:4]])
    A = lambda fc: np.interp(fc, fq, aq.real) + 1j * np.interp(fc, fq, aq.imag)
    out()
    out('line   f [cpd]    h\'2 rotating        h\'2 sphere   dh\'2               dh(OT) ours          '
        'dh(OT) IERS (Table 7.2 - resonance formula)')
    for wave, fc in LINES:
        hr, hs = h_load(md, geo, fc)
        d = fac * A(fc) * (hr - hs)
        iers = complex(etide.table72('h0', fc)) - complex(etide.mhb_resonance('h0', fc))
        out(f'{wave:5s} {fc:.6f}  {complex(hr):+.5f}  {complex(hs).real:+.5f}  {complex(hr - hs):+.5f}  '
            f'{d:+.5f}  {iers:+.5f}')
    Path('results/otl_fcn_resonance.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
