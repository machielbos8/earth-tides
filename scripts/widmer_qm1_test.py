# -*- coding: utf-8 -*-
#
# Q model test: the anelastic Love numbers of Tables S5/S6 with the shear Q of Widmer et al. (1991, QM1).
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""Does the Q model explain why our imaginary parts (Table S6) are 5-16 per cent larger than the Conventions'?

The anelastic values of the IERS Conventions use the shear Q of Widmer, Masters & Gilbert (1991, GJI 104, 541;
model QM1, their Table 3), power law alpha = 0.15 from 200 s (Mathews, Dehant & Gipson 1997). Our tables use the
Q of PREM. Here PREM (and PREMs, the static model of the non-hydrostatic state) is split at the radii of QM1 and
every sub-layer gets the QM1 shear Q at its midpoint; elastic structure unchanged. Simplifications: inner core a
constant Q 110 (QM1: 109-112; Widmer et al.: a constant inner-core Q fits as well), QM1 boundaries within 10 km of a
PREM boundary snapped to it (1229.4 -> ICB, 3484.3 -> CMB, 5700 -> 5701 km), bulk Q ignored (the model has none;
QM1: 2920 in the upper mantle, 12000 in the outer core). Every variant refits the excess e_f to the FCN of 431.4 sd
as in Table S5 (make_table_anelastic.state_at).

Variants: PREM Q unsplit (= Table S5), PREM Q split (control: the splitting alone), QM1 with t0 = 200 s (the IERS
assumption) and t0 = 1 s.

    .venv/bin/python scripts/widmer_qm1_test.py   ->  results/widmer_qm1_test.txt"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper_pepi'))
warnings.filterwarnings('ignore')
if '--figure' not in sys.argv:                  # make_table4 reads the figure from argv; Tables S5/S6 use 'level'
    sys.argv += ['--figure', 'level']           # (also in the spawned workers, which re-import this module)
import numpy as np

# Widmer et al. (1991) Table 3, mantle: (radius of the top of the layer [km], Q_mu)
QM1_MANTLE = ((3656.7, 266.), (3829.9, 268.), (4003.0, 271.), (4176.2, 272.), (4349.3, 274.), (4522.5, 285.),
              (4695.6, 311.), (4868.8, 343.), (5042.0, 360.), (5215.1, 368.), (5353.6, 379.), (5492.2, 383.),
              (5701.0, 362.), (5825.0, 286.), (5950.0, 210.), (6052.5, 152.), (6180.6, 121.), (6283.1, 116.),
              (6371.0, 120.))
QM1_INNER = 110.
SNAP_KM = 10.0
VARIANTS = (('PREM Q', False, 'prem', 200.0), ('PREM Q split', True, 'prem', 200.0),
            ('QM1 t0 200 s', True, 'qm1', 200.0), ('QM1 t0 1 s', True, 'qm1', 1.0))


def q_at(r_km, q_old, fluid):
    if fluid or not np.isfinite(q_old):
        return np.nan
    if r_km < 1300.0:
        return QM1_INNER
    for top, q in QM1_MANTLE:
        if r_km <= top:
            return q
    raise ValueError(r_km)


def patch(qsource):
    """Replace earthtides.model.EarthModel by a subclass that splits every model at the QM1 radii and, for
    qsource 'qm1', sets the QM1 shear Q."""
    from earthtides import model as M
    base = M.EarthModel

    class Split(base):
        def __init__(self, filename):
            super().__init__(filename)
            md = self
            for top, _ in QM1_MANTLE[:-1]:
                r = top * 1e3
                i = int(np.searchsorted(md.r1, r))
                if min(r - md.r0[i], md.r1[i] - r) > SNAP_KM * 1e3:
                    md = md.split_layer(i, [r])
            if qsource == 'qm1':
                q = []
                for i in range(md.n_layers):
                    rm = 0.5 * (md.r0[i] + md.r1[i])
                    q.append(q_at(rm / 1e3, md.Qmu[i], float(md.mu[i](rm)) == 0.0))
                md.Qmu = np.array(q)
            self.__dict__.update(md.__dict__)

    M.EarthModel = Split


def run_variant(v):
    label, split, qsource, t0 = v
    if split:
        patch(qsource)
    import make_table_anelastic as ta
    from earthtides import love
    opts, s = ta.state_at(t0)
    md = love._model(opts)[0]
    layers = [(round(md.r0[i] / 1e3, 1), round(md.r1[i] / 1e3, 1), float(md.Qmu[i])) for i in range(md.n_layers)]
    return label, s, ta.values(opts), layers


def main(run):
    import make_table_anelastic as ta
    with Pool(2, maxtasksperchild=1) as pool:
        res = pool.map(run_variant, VARIANTS, chunksize=1)
    keys = ta.KEYS
    sel = [keys.index(k) for k in ('h0', 'l0', 'k')]
    lines = [f'Q model test (Widmer et al. 1991, QM1); figure {ta.FIGURE}, FCN 431.4 sd, elastic support', '']
    for label, s, _, layers in res:
        lines.append(f'{label:14s} FCN {s["state"]:.2f} sd, e_f {s["e_f"]:.6e}, CMB {s["cmb"]:.0f} m, '
                     f'{len(layers)} layers')
    lines += ['', 'layers of the QM1 variant (r0, r1 [km], Q_mu):']
    lines += [f'  {a:7.1f} {b:7.1f} {q:6.0f}' for a, b, q in res[2][3]]
    for part, ip in (('real', 0), ('imaginary', 1)):
        lines += ['', f'{part} parts: h0, l0, k'.ljust(30) + '  '.join(f'{k:>27s}' for k in ('h0', 'l0', 'k'))]
        for name, m, f in ta.WAVES:
            iers = (ta.iers if ip == 0 else ta.iers_im)(name, m, f)
            for label, _, V, _ in res:
                v = V[name][ip]
                lines.append(f'{name:10s} {label:18s} ' + ' '.join(f'{v[i]:9.5f}' for i in sel))
            lines.append(f'{name:10s} {"IERS":18s} ' + ' '.join(' ' * 9 if iers[i] is None else f'{iers[i]:9.5f}'
                                                              for i in sel))
    # ratios of the imaginary parts to the Conventions' (away from the resonance)
    lines += ['', 'Im h0 / Im h0 (IERS), Im l0 / Im l0 (IERS):']
    for name, m, f in ta.WAVES:
        iers = ta.iers_im(name, m, f)
        cells = []
        for label, _, V, _ in res:
            v = V[name][1]
            cells.append(f'{label}: ' + ' '.join(f'{v[i] / iers[i]:6.3f}' if iers[i] else '   -  '
                                                  for i in (keys.index('h0'), keys.index('l0'))))
        lines.append(f'{name:10s} ' + ' | '.join(cells))
    run.write_text(str(ROOT / 'results' / 'widmer_qm1_test.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    from provenance import Run
    with Run('Q model test: Tables S5/S6 with the shear Q of Widmer et al. (1991, QM1)',
             params=dict(figure=sys.argv[sys.argv.index('--figure') + 1], variants=[v[0] for v in VARIANTS], snap_km=SNAP_KM, qm1_inner=QM1_INNER),
             inputs=['data/PREM.dat', 'data/PREMs.dat']) as run_:
        main(run_)
