# -*- coding: utf-8 -*-
#
# Fig. S1 of the P1 supplement (PEPI): published Love numbers minus ours, elastic and anelastic.
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
"""The two comparison tables of the supplement as one figure: for each parameter and tide, a published value
minus ours, in units of 1e-4.
  elastic, hydrostatic PREM (Table S2, paper_pepi/table_ddw.tex, first-order figure, like for like):
      Dehant et al. (1999) minus ours;
  anelastic, non-hydrostatic PREM with the FCN at 431.4 sd (results/paper1_table_anelastic_level.txt, the data
  of Table tab:ddw_anelastic, make_table_anelastic.py):
      Dehant et al. (1999) minus ours, IERS Conventions minus ours, ours with the IERS dispersion minus ours.
The elastic numbers are read from Table S2 of the supplement (paper_pepi/table_ddw.tex), which is their record (they combine the
viscous limit of several runs; see the lab notes of 2026-11-15). psi1, where the resonance makes the
differences 1e-2, is drawn at the edge of the panel (arrow) and its values are in the tables.

    .venv/bin/python paper_pepi/make_fig_ddw_iers.py
    ->  paper_pepi/figures/figS_ddw_iers.pdf"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper'))
import numpy as np

from provenance import Run

SUPP = ROOT / 'paper_pepi' / 'table_ddw.tex'          # Table S2 of the supplement
ANEL = ROOT / 'results' / 'paper1_table_anelastic_level.txt'
OUT = ROOT / 'paper_pepi' / 'figures' / 'figS_ddw_iers.pdf'
TIDES = ('Mf', 'O$_1$', 'P$_1$', 'K$_1$', r'$\psi_1$', r'$\phi_1$', 'OO$_1$', 'M$_2$')
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')
PANELS = (('h0', '$h^{(0)}$'), ('l0', '$l^{(0)}$'), ('k', '$k$'),
          ('h2', '$h^{(2)}$'), ('l2', '$l^{(2)}$'), ('l1', '$l^{(1)}$'))
SKIP_SCALE = r'$\psi_1$'                     # off scale: drawn as an arrow at the edge


def num(s):
    s = s.strip().strip('$').replace(' ', '')
    return float(s) if s else None


def elastic():
    """Ours and DDW from Table tab:ddw of the supplement: {tide: {'ours': (...), 'DDW': (...)}}."""
    t = SUPP.read_text()
    t = t[t.index(r'\label{tab:ddw}'):]
    t = t[:t.index(r'\midrule', t.index(r'\midrule') + 1)]
    rows, cur = {}, None
    for line in t.splitlines():
        if '&' not in line or 'tide &' in line:
            continue
        c = [x.strip() for x in line.rstrip('\\ ').split('&')]
        name = 'Mf' if c[0] == 'zonal' else c[0]
        if name:
            cur = name
            rows[cur] = {'ours': tuple(num(x) for x in c[2:8])}
        elif c[1] == 'DDW':
            rows[cur]['DDW'] = tuple(num(x) for x in c[2:8])
    return rows


def anelastic():
    """{tide: {line: (h0, h2, l0, l1, l2, k)}} from the text output of make_table_anelastic.py."""
    rows = {}
    for line in ANEL.read_text().splitlines():
        m = re.match(r'^(\S+)\s+(this paper|IERS dispersion|DDW|IERS)\s+(.*)$', line)
        if not m:
            continue
        rest = line[28:]                       # fixed width: name 10, line 16, then six columns of 9 + 1
        vals = [rest[i:i + 10].strip() for i in range(0, 60, 10)]
        rows.setdefault(m.group(1), {})[m.group(2)] = tuple(float(v) if v else None for v in vals)
    return rows


def diff(a, b, key):
    i = KEYS.index(key)
    return None if a is None or b is None or a[i] is None or b[i] is None else 1e4 * (a[i] - b[i])


def plot(E, A):
    import make_figures as mf
    plt = mf.plt
    series = (  # label, colour, marker, filled, (tide -> value)
        ('elastic: Dehant et al. (1999) $-$ ours', mf.C1, 'o', False,
         lambda t, k: diff(E[t]['DDW'], E[t]['ours'], k)),
        ('anelastic: Dehant et al. (1999) $-$ ours', mf.C1, 'o', True,
         lambda t, k: diff(A[t]['DDW'], A[t]['this paper'], k)),
        ('anelastic: IERS Conventions $-$ ours', mf.C2, 's', True,
         lambda t, k: diff(A[t]['IERS'], A[t]['this paper'], k)),
        ('anelastic: ours with the IERS dispersion $-$ ours', mf.C3, 'D', True,
         lambda t, k: diff(A[t]['IERS dispersion'], A[t]['this paper'], k)))
    dx = np.linspace(-0.27, 0.27, len(series))
    fig, axs = plt.subplots(2, 3, figsize=(mf.W2COL, 4.3))
    for ax, (key, lab) in zip(axs.flat, PANELS):
        vals = [[s[4](t, key) for t in TIDES] for s in series]
        scale = [v for vs in vals for t, v in zip(TIDES, vs) if v is not None and t != SKIP_SCALE]
        lo, hi = min(scale + [-1.0]), max(scale + [1.0])
        pad = 0.12 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.axhspan(-1, 1, color=mf.MUTED, alpha=0.18, lw=0)
        ax.axhline(0, color=mf.MUTED, lw=0.5)
        for (slab, col, mk, filled, _), vs, d in zip(series, vals, dx):
            for i, v in enumerate(vs):
                if v is None:
                    continue
                x = i + d
                if v > hi or v < lo:                 # off scale: on the frame, so no value is read from the axis
                    ax.plot(x, hi if v > hi else lo, '^' if v > hi else 'v', color=col, ms=5, mew=0.8,
                            mfc=col if filled else 'white', clip_on=False, zorder=5)
                    continue
                ax.plot(x, v, mk, color=col, ms=3.5, mew=0.8, mfc=col if filled else 'none')
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.6, len(TIDES) - 0.4)
        ax.set_xticks(range(len(TIDES)))
        ax.set_xticklabels([t.replace('$', '').replace('_', '').replace('\\psi', 'ψ').replace('\\phi', 'φ')
                            for t in TIDES], fontsize=6.5)
        for i in range(1, len(TIDES)):
            ax.axvline(i - 0.5, color=mf.MUTED, lw=0.3, alpha=0.5)
        ax.set_title(lab, fontsize=8, loc='left')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'difference ($10^{-4}$)')
    h = [plt.Line2D([], [], ls='', marker=s[2], color=s[1], mfc=s[1] if s[3] else 'none', ms=4) for s in series]
    fig.legend(h, [s[0] for s in series], loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT)
    return OUT


def main():
    with Run('Fig. S1 of the P1 supplement: Dehant et al. (1999) and IERS Love numbers minus ours',
             params=dict(units=1e-4), inputs=[str(SUPP.relative_to(ROOT)), str(ANEL.relative_to(ROOT))]) as run:
        E, A = elastic(), anelastic()
        for t in TIDES:
            print(f'{t:10s}', ' '.join(f'{k}:{diff(E[t]["DDW"], E[t]["ours"], k) or 0:+7.1f}' for k in KEYS))
        out = plot(E, A)
        run.output(str(out.relative_to(ROOT)))


if __name__ == '__main__':
    main()
