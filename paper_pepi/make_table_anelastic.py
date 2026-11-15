# -*- coding: utf-8 -*-
#
# Table S4b of P1 (PEPI): anelastic, non-hydrostatic Love numbers against Dehant et al. (1999) and the IERS.
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""The anelastic counterpart of Table S4 (tab:ddw), four lines per tide:
 1. this paper: anelastic PREM (constant Q from 1 s to 200 s, power law with alpha = 0.15 beyond; the model of
    Sect. 2), elastic support of the excess J2 of Chambat et al. (2010) and the excess e_f that puts the FCN at
    431.4 sidereal days (the state of Table 4, column 'FCN at 431 days');
 2. the same with the IERS assumption, power law from 200 s only (love.Options(t0=200): no constant-Q band), its
    excess e_f refitted to the same FCN;
 3. Dehant et al. (1999), inelastic non-hydrostatic Earth, tables 11 and 12 (FCN 431.4 sd as well);
 4. IERS Conventions (Petit & Luzum 2010): diurnal from the resonance formula (Table 7.1; without the ocean-tide
    term of Table 7.2), zonal from eqs. 7.4a,b with the band constants h(2) = -0.0006, l(2) = 0.0002,
    semidiurnal the nominal values (h(0) = 0.6078, l(0) = 0.0847, h(2), l(2), l(1) = 0.0024); no k (the
    Conventions give k in their Chapter 6, which the paper does not use).
Real parts. Figure of EARTHTIDES_FIGURE / --figure (the paper: level), slipping core with the term of a non-level
wall, K 4 (love module defaults), IERS conventions of the parametrization.

    .venv/bin/python paper_pepi/make_table_anelastic.py --figure level
    ->  paper_pepi/table_anelastic.tex, results/paper1_table_anelastic[_level].txt"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper_pepi'))
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love
from earthtides import etide
import make_table4 as t4                        # fcn(), J2, E_MHB, FIGURE, SUF

FIGURE, SUF = t4.FIGURE, t4.SUF
P_OBS = 431.4
SID = 1.002737909                               # cycles per day per cycle per sidereal day
# (name in the table, m, frequency [cpd]) as in Table 4; cpsd in brackets: Mf 0.0730, O1 0.92700, ...
WAVES = (('Mf', 0, 0.07320), ('O$_1$', 1, 0.92954), ('P$_1$', 1, 0.99726), ('K$_1$', 1, 1.00274),
         (r'$\psi_1$', 1, 1.00548), (r'$\phi_1$', 1, 1.00821), ('OO$_1$', 1, 1.07594), ('M$_2$', 2, 1.93227))
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')
# Dehant et al. (1999), inelastic non-hydrostatic: table 11 (h0, h2, l0, l1, l2; Mf has no l1) and table 12 (k0)
DDW = {'Mf': (0.61346, -0.00050, 0.08657, None, 0.00008, 0.30285),
       'O$_1$': (0.60466, -0.00050, 0.08539, 0.00073, 0.00008, 0.29853),
       'P$_1$': (0.58206, -0.00054, 0.08617, 0.00071, 0.00009, 0.28713),
       'K$_1$': (0.52227, -0.00067, 0.08821, 0.00061, 0.00011, 0.25697),
       r'$\psi_1$': (1.08671, 0.00052, 0.06901, 0.00155, -0.00011, 0.54166),
       r'$\phi_1$': (0.67032, -0.00036, 0.08317, 0.00086, 0.00005, 0.33164),
       'OO$_1$': (0.61043, -0.00048, 0.08522, 0.00078, 0.00008, 0.30143),
       'M$_2$': (0.61042, -0.00048, 0.08577, 0.00097, 0.00010, 0.30235)}


def iers(name, m, f):
    if m == 1:
        return tuple(float(etide.mhb_resonance(k, f).real) for k in ('h0', 'h2', 'l0', 'l1', 'l2')) + (None,)
    if m == 0:
        return (float(etide.zonal_anelastic('h0', f).real), -0.0006, float(etide.zonal_anelastic('l0', f).real),
                None, 0.0002, None)
    return (0.6078, -0.0006, 0.0847, 0.0024, 0.0002, None)


def state_at(t0):
    """Options of the elastically supported state whose FCN is P_OBS, for reference period t0; FCNs."""
    hyd = love.Options(figure=FIGURE, t0=t0)
    c0 = t4.fcn(hyd)
    m1 = love.Options(figure=FIGURE, t0=t0, nonhydrostatic=dict(targets={'J2': t4.J2, 'e_f': t4.E_MHB}))
    c1 = t4.fcn(m1)
    e = float(t4.E_MHB * (1 / P_OBS - (c0 - 1)) / ((c1 - 1) - (c0 - 1)))
    m2 = love.Options(figure=FIGURE, t0=t0, nonhydrostatic=dict(targets={'J2': t4.J2, 'e_f': e}))
    c2 = t4.fcn(m2)
    cmb = love._model(m2)[2].summary()['response']['cmb']
    return m2, dict(hydrostatic=1 / (c0 - 1), state=1 / (c2 - 1), e_f=e, cmb=cmb)


def values(opts):
    out = {}
    for name, m, f in WAVES:
        p = love._job((opts, 2, m, f))
        out[name] = tuple(float(complex(p[k]).real) if k in p and not (m == 0 and k == 'l1') else None
                          for k in KEYS)
    return out


def fmt(v):
    return '' if v is None else (f'${v:.5f}$' if v < 0 else f'{v:.5f}')


def main(run):
    o1, s1 = state_at(1.0)
    o2, s2 = state_at(200.0)
    V1, V2 = values(o1), values(o2)
    rows = {name: [('this paper', V1[name]), ('IERS dispersion', V2[name]), ('DDW', DDW[name]),
                   ('IERS', iers(name, m, f))] for name, m, f in WAVES}
    lines = [f'figure {FIGURE}; FCN {P_OBS} sd target', f'this paper (t0 1 s): {s1}', f'IERS dispersion (t0 200 s): {s2}',
             '', f'{"tide":10s} {"line":16s} ' + ' '.join(f'{k:>9s}' for k in KEYS)]
    for name, _, _ in WAVES:
        for lab, v in rows[name]:
            lines.append(f'{name:10s} {lab:16s} ' + ' '.join(' ' * 9 if x is None else f'{x:9.5f}' for x in v))
    run.write_text(str(ROOT / 'results' / f'paper1_table_anelastic{SUF}.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    cap = (r'Degree-2 Love numbers of the anelastic, non-hydrostatic Earth in the definition of '
           r'\citet{mathews1995} (real parts), each with the free core nutation at 431.4 sidereal days. This paper: '
           r'anelastic PREM of Section~\ref{M-sec:anelastic} (constant $Q_\mu$ from 1 to 200~s, $\omega^{0.15}$ '
           r'beyond) on the level figure, elastic support of the excess $J_2$ of \citet{chambat2010} and %.0f~m of '
           r'excess flattening of the core--mantle boundary (Table~\ref{tab:nonhydro_love}); IERS dispersion: the '
           r'same model with the power law from 200~s only, the assumption behind the anelastic values of the '
           r'Conventions \citep{mathews1997} (%.0f~m for the same period); DDW: \citet[][tables 11 and 12]{dehant1999}, '
           r'inelastic, non-hydrostatic; IERS: \citet{petit2010}, diurnal from the resonance formula (their '
           r'Table~7.1, without the ocean-tide term of their Table~7.2), zonal from their eqs.~7.4a,b, '
           r'semidiurnal the nominal values, latitude terms outside the diurnal band constant; the Conventions '
           r'give $k$ in another chapter.' % (s1['cmb'], s2['cmb']))
    tex = [r'\begin{table*}', r'\caption{' + cap + '}', r'\label{tab:ddw_anelastic}',
           r'\small\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}', r'\toprule',
           r'tide & & $h^{(0)}$ & $h^{(2)}$ & $l^{(0)}$ & $l^{(1)}$ & $l^{(2)}$ & $k$ \\', r'\midrule']
    for i, (name, _, _) in enumerate(WAVES):
        if i:
            tex.append(r'\addlinespace[2pt]')
        for j, (lab, v) in enumerate(rows[name]):
            tex.append((name if j == 0 else '') + f' & {lab} & ' + ' & '.join(fmt(x) for x in v) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    run.write_text(str(ROOT / 'paper_pepi' / 'table_anelastic.tex'), '\n'.join(tex) + '\n')


if __name__ == '__main__':
    from provenance import Run
    with Run('anelastic non-hydrostatic Love numbers against DDW 1999 and the IERS (P1 Table S4b)',
             params=dict(figure=FIGURE, fcn=P_OBS, t0=[1.0, 200.0]), inputs=['data/PREM.dat']) as run_:
        main(run_)
