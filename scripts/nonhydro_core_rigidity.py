# -*- coding: utf-8 -*-
#
# Does the non-hydrostatic state depend on the rigidity of the weak-solid core of its static solve?
# (P1 v2 Sect. 6.2)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""A static liquid core is undetermined (Longman 1963), so the static support of the non-hydrostatic state
uses PREMs, whose outer core is a weak solid with Vs = 1 m/s. This script rebuilds the elastic state with
the excess J2 of Chambat et al. (2010) and the e_f of Mathews et al. (2002), as Table S10 (make_table4.py),
with the core Vs of the static model set to VS (m/s), and gives per case the fitted amplitudes, the targets
reached, the FCN period and the anelastic Love numbers of Table S10, as differences from Vs = 1 m/s.
The tidal solves use the true liquid core of PREM in every case.
Usage: python scripts/nonhydro_core_rigidity.py  ->  results/nonhydro_core_rigidity.txt"""
import sys
import tempfile
import warnings
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'paper_pepi')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love
import make_table4 as t4

VS = (1.0, 0.3, 3.0, 10.0)                  # m/s; 1 = PREMs as used
SRC = Path('data/PREMs.dat')
CORE_ROW = '5149.5    0.00100'              # outer-core rows of the Vs_h and Vs_v blocks


def variant(vs, tmp):
    """PREMs with the outer-core Vs replaced (both Vs blocks)."""
    txt = SRC.read_text()
    assert txt.count(CORE_ROW) == 2, 'unexpected PREMs.dat layout'
    p = Path(tmp) / f'PREMs_vs{vs:g}.dat'
    p.write_text(txt.replace(CORE_ROW, f'5149.5    {vs / 1e3:.5f}'))
    return str(p)


def main():
    from provenance import Run
    with Run('weak-solid core rigidity of the static support: sensitivity of the non-hydrostatic state',
             params=dict(vs=VS, J2=t4.J2, e_f=t4.E_MHB), inputs=[str(SRC), 'data/PREM.dat']) as run_, \
            tempfile.TemporaryDirectory() as tmp:
        hyd = love.Options()
        T0 = t4.table(hyd)
        res = {}
        for vs in VS:
            o = love.Options(nonhydrostatic=dict(targets={'J2': t4.J2, 'e_f': t4.E_MHB}),
                             model_static=variant(vs, tmp))
            st = love._model(o)[2]
            res[vs] = (st.summary(), 1 / (t4.fcn(o) - 1), t4.table(o))
            print(f'Vs {vs:g} m/s: amplitudes {st.summary()["amplitudes"]}, FCN {res[vs][1]:.2f} sd', flush=True)
        ref = res[1.0]
        lines = [f'non-hydrostatic elastic state (J2 {t4.J2:.3g}, e_f {t4.E_MHB:.3g}) with the core Vs of the static '
                 f'model varied; tidal solves with the liquid core of PREM',
                 f'{"Vs m/s":>7s} {"rigidity Pa":>11s} {"amplitudes":>30s} {"FCN sd":>8s}  targets reached']
        for vs in VS:
            s, p, _ = res[vs]
            mu = 1.1e4 * vs ** 2                             # rho ~ 9.9-12.2 t/m3 in the outer core
            lines.append(f'{vs:7g} {mu:11.1e} {" ".join(f"{a:+.6e}" for a in s["amplitudes"]):>30s} {p:8.2f}  '
                         + ', '.join(f'{k} {v:.6g}' for k, v in s['response'].items()))
        lines += ['', 'non-hydrostatic minus hydrostatic Love numbers (1e-4, real parts) at Vs = 1 m/s, and the '
                  'change of that difference for the other Vs (1e-6)',
                  f'{"wave":8s} ' + ' '.join(f'{k:>7s}' for k, _ in t4.KEYS) + '   | '
                  + ' | '.join(f'Vs {vs:g}: ' + ' '.join(f'{k:>6s}' for k, _ in t4.KEYS) for vs in VS[1:])]
        for name, _, _ in t4.WAVES:
            h, r = T0[name], ref[2][name]
            row = f'{name:8s} ' + ' '.join(f'{1e4 * (r[k].real - h[k].real):+7.2f}' if k in h else ' ' * 7
                                           for k, _ in t4.KEYS)
            for vs in VS[1:]:
                t = res[vs][2][name]
                row += '   | ' + ' ' * (len(f'Vs {vs:g}: ')) + ' '.join(
                    f'{1e6 * (t[k].real - r[k].real):+6.2f}' if k in h else ' ' * 6 for k, _ in t4.KEYS)
            lines.append(row)
        run_.write_text('results/nonhydro_core_rigidity.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
