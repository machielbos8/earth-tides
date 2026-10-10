# -*- coding: utf-8 -*-
#
# Observed K1/O1 tidal gravity ratios (Ducarme et al. 2009, 2014) against our gravimetric factors:
# the sign of the in-phase resonant ocean-loading correction.
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
"""rho = [dc(K1)/dc(O1)]_observed / [delta(K1)/delta(O1)]_theory - 1.

Observations: corrected amplitude factors dc (same series for O1 and K1; ocean loading from the
mean of 9 ocean tide models with Green's functions, i.e. without the FCN resonance of the load):
Ducarme et al. (2009), Table 5 (7 GGP stations and Pecny); Ducarme et al. (2014), Table 1 (Pecny
OSG050, Vienna, Conrad; mean of 8 models) and Table 3 (Strasbourg). The ratio removes the
calibration. Theory: this paper, anelastic hydrostatic PREM with the FCN at 431 sidereal days
(results/gravimetric_factor_spinup.txt), and the published DDW99/NH and MAT01/NH values (2014,
Table 1). Predicted rho from the resonant load term: +0.064 % with our admittance, -0.064 % with
the empirical admittance of the IERS Conventions; phase +0.038 deg in both.
Usage: python scripts/check_tidal_gravity_ducarme.py -> results/tidal_gravity/check_ducarme.txt
"""
from pathlib import Path

import numpy as np

OURS = {'Membach': 0.983004, 'Strasbourg': 0.983009, 'Bad Homburg': 0.983005, 'Moxa': 0.983004,
        'Medicina': 0.983020, 'Wettzell': 0.983008, 'Vienna': 0.983010, 'Pecny': 0.983006,
        'Conrad': 0.983011}
OTHER = {'DDW99/NH': 1.13451 / 1.15429, 'MAT01/NH': 1.13495 / 1.15402}
# (dc O1, dc K1, source)
OBS = [('Membach', 1.15359, 1.13568, '2009'), ('Strasbourg', 1.15366, 1.13547, '2009'),
       ('Bad Homburg', 1.15450, 1.13600, '2009'), ('Moxa', 1.15315, 1.13507, '2009'),
       ('Medicina', 1.15334, 1.13480, '2009'), ('Wettzell', 1.15237, 1.13424, '2009'),
       ('Vienna', 1.15224, 1.13404, '2009'), ('Pecny', 1.15443, 1.13672, '2009'),
       ('Pecny', 1.15373, 1.13551, '2014'), ('Vienna', 1.15337, 1.13507, '2014'),
       ('Conrad', 1.15340, 1.13511, '2014'), ('Strasbourg', 1.15308, 1.13477, '2014')]
PRED = 0.064
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def main():
    out('rho [%] = observed K1/O1 over theoretical K1/O1, minus 1 (in-phase amplitude factors)')
    out(f'predicted: +{PRED} (our admittance), -{PRED} (empirical IERS admittance)')
    out(f"{'station':12s} {'paper':5s} {'K1/O1 obs':>10s} {'ours':>8s} {'DDW99/NH':>9s} {'MAT01/NH':>9s}")
    rho = {k: [] for k in ('ours', 'DDW99/NH', 'MAT01/NH')}
    for st, o1, k1, src in OBS:
        r = k1 / o1
        v = dict(ours=100 * (r / OURS[st] - 1),
                 **{k: 100 * (r / th - 1) for k, th in OTHER.items()})
        for k in rho:
            rho[k].append(v[k])
        out(f"{st:12s} {src:5s} {r:10.6f} {v['ours']:+8.3f} {v['DDW99/NH']:+9.3f} {v['MAT01/NH']:+9.3f}")
    out()
    for k, v in rho.items():
        v = np.array(v)
        m, sd = v.mean(), v.std(ddof=1)
        se = sd / np.sqrt(len(v))
        out(f'{k:9s} mean {m:+.3f} %  sd {sd:.3f}  se {se:.3f}   '
            f'distance to our sign {(m - PRED) / se:+.1f} se, to IERS sign {(m + PRED) / se:+.1f} se')
    Path('results/tidal_gravity/check_ducarme.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
