# -*- coding: utf-8 -*-
#
# Figures of paper 1 (PEPI) that are new: the supported non-hydrostatic state and the FCN period
# against the excess dynamical ellipticity of the core.
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""New figures of paper 1 (PEPI). Run from the project root:

    .venv/bin/python paper_pepi/make_figures.py          # compute (cached) + plot
    .venv/bin/python paper_pepi/make_figures.py --replot # plot from the cache only

fig_nonhydro_state  the non-hydrostatic state of elastic PREM fitted to an excess J2 of 1.14e-5
                    (Chambat et al. 2010) and 450 m of excess CMB flattening, supported elastically
                    and by viscous flow: (a) excess equatorial-minus-polar radius of the coordinate
                    surfaces (material surfaces; interpolated between anchors in the viscous mantle),
                    (b) the deviatoric stress that holds the shape (largest over latitude).
figS_nonhydro_state_viscous  the same for viscous support (Online Resource)
fig_fcn_ef          FCN period against the excess dynamical ellipticity of the core: consistent
                    state (stress included), the route without the stress, and the formula
                    (A/A_m)(e_f - beta); slipping inviscid core from results/nonhydro_diurnal.txt, viscous
                    (no-slip) core from results/nonhydro_fcn_viscous_*_K10_N30.txt (delta -> 0).
"""
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import figure_choice as fc          # EARTHTIDES_FIGURE=level: inputs and cache with _level, the level figure
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = ROOT / 'paper_pepi' / 'figures'
CACHE = Path(fc.tagged(ROOT / 'results' / 'figdata_nonhydro.npz'))
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#8a8984'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'legend.frameon': False, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W1COL, W2COL = 3.5, 7.2           # Elsevier single column / full width (inches)
TARGETS = {'J2': 1.14e-5, 'cmb': 450.0}
E_F_HYD, BETA = 0.0025452, 6.07e-4
if fc.FIGURE == 'level':            # e_f of the mapped core (nonhydro_diurnal_level.txt); beta from its FCN, A_m / A = 0.8862
    _t = (ROOT / 'results' / 'nonhydro_diurnal_level.txt').read_text()
    E_F_HYD = float(re.search(r'e_f of the hydrostatic core ([\d.]+)', _t).group(1))
    BETA = E_F_HYD - 0.8862 / float(re.search(r'hydrostatic\s+FCN\s+([\d.]+)', _t).group(1))


def compute(run):
    from earthtides.model import EarthModel
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    from earthtides.nonhydro import NonHydrostatic
    md, mds = EarthModel(ROOT / 'data/PREM.dat'), EarthModel(ROOT / 'data/PREMs.dat')
    base = fc.base(md)
    x, w = np.polynomial.legendre.leggauss(32)
    th, P2 = np.arccos(x), 0.5 * (3 * x * x - 1)
    d = {}
    for kind in ('elastic', 'viscous'):
        nh = NonHydrostatic(md, dict(support=kind, targets=TARGETS), base, OMEGA_EARTH, mds)
        st = nh.state
        R, D, S, L = [], [], [], []
        for i in range(md.n_layers):
            s = np.linspace(max(md.r0[i], 2e4), md.r1[i], 25)
            # core: deflection of the level surfaces (the map adds a gauge term there)
            eta = st._zeta(i, s, th) if i == st.ioc else st.eta2(i, s, th)
            R.append(s)
            L.append(np.full(len(s), i))
            D.append(-1.5 * 2.5 * (eta * (w * P2)[None, :]).sum(axis=1))      # eq - pole radius
            T = st.tau(i, s[:, None], th, {'R': 0 * s[:, None] + 0 * th[None, :]})
            if T is None:
                S.append(0 * s)
                continue
            rr, tt, pp, rt = (np.broadcast_to(T[k].v, (len(s), len(th))) for k in ('rr', 'tt', 'pp', 'rt'))
            m = (rr + tt + pp) / 3
            j2 = 0.5 * ((rr - m)**2 + (tt - m)**2 + (pp - m)**2) + rt**2
            S.append(np.sqrt(j2).max(axis=1))
        d[kind + '_r'] = np.concatenate(R)
        d[kind + '_defl'] = np.concatenate(D)
        d[kind + '_stress'] = np.concatenate(S)
        d[kind + '_layer'] = np.concatenate(L)
        d[kind + '_surface'] = nh.response['surface']
    d['interfaces'] = np.array([md.r1[i] for i in range(md.n_layers - 1)])
    run.savez(str(CACHE), **d)


def fig_nonhydro_state(kind='elastic', name='fig_nonhydro_state'):
    """(a) excess equatorial-minus-polar radius of the material surfaces (core: level surfaces),
    (b) the deviatoric stress (largest over latitude), per layer."""
    F = dict(np.load(CACHE))
    fig, axs = plt.subplots(1, 2, figsize=(W2COL, 2.6), sharey=True)
    r, lay = F[kind + '_r'] / 1e3, F[kind + '_layer']
    for j in np.unique(lay):
        k = lay == j
        c = C2 if 1221.5 < np.median(r[k]) < 3480.0 else C1      # liquid core: level surfaces
        axs[0].plot(F[kind + '_defl'][k], r[k], color=c)
        axs[1].plot(F[kind + '_stress'][k] / 1e6, r[k], color=c)
        if c == C2:
            i = np.argmin(abs(r[k] - 2500.0))
            axs[0].annotate('liquid core:\nlevel surfaces', (F[kind + '_defl'][k][i], r[k][i]), color=C2,
                            fontsize=6.5, textcoords='offset points', xytext=(8, 0), va='center')
    for ax in axs:
        for rb in (1221.5, 3480.0, 5701.0):
            ax.axhline(rb, color=MUTED, lw=0.5, ls=':')
        ax.set_ylim(0, 6700)                                # the surface (6371 km) well inside the frame
    xr = axs[0].get_xlim()[1]
    for rb, lab in ((1221.5, 'ICB'), (3480.0, 'CMB'), (5701.0, '670 km')):
        axs[0].text(xr, rb + 60, lab, color=MUTED, fontsize=6.5, ha='right')
    axs[0].set_xlabel('excess equatorial minus polar radius (m)')
    axs[0].set_ylabel('radius (km)')
    axs[1].set_xlabel('deviatoric stress (MPa)')
    axs[0].set_title(f'(a) {kind} support', loc='left', fontsize=8)
    axs[1].set_title('(b) stress that holds the shape', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.pdf')


def read_diurnal():
    """CMB excess, e_f excess (%), FCN of the consistent state and of the route without stress."""
    txt = Path(fc.tagged(ROOT / 'results' / 'nonhydro_diurnal.txt')).read_text()
    c0 = float(re.search(r'hydrostatic\s+FCN\s+([\d.]+)', txt).group(1))
    rows = []
    for block in txt.split('-- CMB excess')[1:]:
        cmb = float(block.split()[0])
        ef = float(re.search(r'e_f \+?(-?[\d.]+) %', block).group(1))
        a = float(re.search(r'A \(DDW route\)\s+FCN\s+([\d.]+)', block).group(1))
        b = float(re.search(r'B \(consistent\)\s+FCN\s+([\d.]+)', block).group(1))
        # without the deviatoric stress = route D of nonhydro_stress_split.py: density and the slip terms of the
        # non-level boundary, no tau_0 (2026-11-15; route A also drops the slip term of a wall off level, the
        # boundary condition of Sect. 5.3, and is not a state 'without the deviatoric stress')
        sp = Path(fc.tagged(ROOT / 'results' / f'nonhydro_stress_split_{cmb:.0f}m.txt'))
        if sp.exists():
            a = float(re.search(r'D: slip terms, no tau\s+([\d.]+)', sp.read_text()).group(1))
        rows.append((cmb, ef, b, a))
    return c0, np.array(rows)


def read_viscous(R):
    """Viscous (no-slip) core, delta -> 0: mean of the linear and quadratic extrapolations of
    scripts/nonhydro_fcn_viscous.py (K 10, N 30, core dp static); rows (e_f excess %, stress, geometry only)."""
    rows = []
    for cmb, ef, _, _ in R:
        p = []
        for case in (f'{cmb:.0f}m_B_cs_static', f'{cmb:.0f}m_A'):
            if fc.FIGURE == 'level':     # K 12, delta 1 .. 0.35 km, quadratic extrapolation (hydrostatic: 456.2 sd)
                txt = (ROOT / 'results' / f'nonhydro_fcn_viscous_{case}_K12_N30_level.txt').read_text()
                p.append(float(re.search(r'delta -> 0, degree 2 .*FCN\s+([\d.]+) sd', txt).group(1)))
                continue
            txt = Path(fc.tagged(ROOT / 'results' / f'nonhydro_fcn_viscous_{case}_K10_N30.txt')).read_text()
            p.append(np.mean([float(v) for v in re.findall(r'delta -> 0, degree \d .*FCN\s+([\d.]+) sd', txt)]))
        rows.append((ef, *p))
    return np.array(rows)


def fig_fcn_ef():
    c0, R = read_diurnal()
    V = read_viscous(R)
    x = np.linspace(-0.5, 6.0, 100)                         # e_f excess, per cent
    off0 = 1 / c0
    period = 1 / (off0 * (E_F_HYD * (1 + x / 100) - BETA) / (E_F_HYD - BETA))
    fig, ax = plt.subplots(figsize=(W1COL, 2.6))
    ax.plot(x, period, color=MUTED, lw=0.8, label=r'eq. (14), $\beta$ fixed')
    # same states, same e_f: filled symbols first, the viscous one smaller with a white rim, hollow ones on top
    ax.plot(R[:, 1], R[:, 2], 'o', color=C1, ms=5.5, label='deviatoric stress, slipping core')
    ax.plot(V[:, 0], V[:, 1], '^', color=C1, mec='white', mew=0.6, ms=4.5, label=r'deviatoric stress, $\delta\rightarrow0$')
    ax.plot(R[:, 1], R[:, 3], 'o', mfc='none', mec=C2, ms=5.5, label='no deviatoric stress, slipping core')
    ax.plot(V[:, 0], V[:, 2], 'v', mfc='none', mec=C2, ms=4.5, label=r'no deviatoric stress, $\delta\rightarrow0$')
    for cmb, ef, b, a in R:
        ax.annotate(f'{cmb:.0f} m', (ef, b), textcoords='offset points', xytext=(5, 0 if cmb == 500 else 3), fontsize=6,
                    color=MUTED)
    ax.axhline(431.4, color=C3, lw=0.6, ls='--')
    ax.text(-0.35, 432.6, 'VLBI', color=C3, fontsize=6.5, ha='left')
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(422, 470)                                   # room for the legend above the points
    ax.set_xlabel(r'excess dynamical ellipticity of the core, $\Delta e_f/e_f$ (%)')
    ax.set_ylabel('FCN period (sidereal days)')
    ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_fcn_ef.pdf')


def fig_am_sweep():
    """Angular-momentum balance through the diurnal band (scripts/nonhydro_am_sweep.py, 450 m)."""
    d = np.load(fc.tagged(ROOT / 'results' / 'nonhydro_am_sweep_450m.npz'))
    x = d['sig']
    fig, ax = plt.subplots(figsize=(W1COL, 2.7))
    ax.semilogy(x, d['am_D'] if 'am_D' in d.files else d['am_A'], '-', color=C2,
                label='non-hydrostatic, no deviatoric stress')        # route D (slip terms kept), 2026-11-15
    ax.semilogy(x, d['am_B'], '-', color=C1, label='non-hydrostatic, with deviatoric stress')
    ax.semilogy(x, d['am_hydrostatic'], ':', color=INK, lw=1.0, label='hydrostatic PREM')
    if fc.FIGURE == 'level':        # the same hydrostatic Earth on the first-order figure, for contrast
        d1 = np.load(ROOT / 'results' / 'nonhydro_am_sweep_450m.npz')
        ax.semilogy(d1['sig'], d1['am_hydrostatic'], '--', color=MUTED, lw=0.8, label='hydrostatic PREM, first-order figure')
    for name, f in (('O$_1$', 0.92700), ('K$_1$', 1.0), ('OO$_1$', 1.07546)):
        ax.axvline(f, color='#d9d8d3', lw=0.6, zorder=0)
        ax.text(f, 1.01, name, ha='center', va='bottom', fontsize=6.5, color=INK, transform=ax.get_xaxis_transform())
    ax.set_xlim(0.88, 1.08)
    ax.set_ylim(*((3e-10, 3e4) if fc.FIGURE == 'level' else (3e-4, 20)))    # legend above the FCN spike
    ax.set_xlabel('frequency (cycles per sidereal day)')
    ax.set_ylabel(r'$|\dot{\mathbf{H}}+\boldsymbol{\Omega}\times\mathbf{H}-\mathbf{N}|\,/\,|\mathbf{N}|$')
    ax.legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_am_sweep.pdf')


def summary(run):
    """Numbers quoted in the text: Sect. 6.2 (sinking of the level surfaces at the top of the core, the
    largest deviatoric stress above the CMB) and Table S10 (FCN periods: formula, slipping core from
    results/nonhydro_diurnal.txt, viscous core as the mean of the two extrapolations)."""
    F = dict(np.load(CACHE))
    lines = []
    for kind in ('elastic', 'viscous'):
        r, lay = F[kind + '_r'], F[kind + '_layer']
        core = np.unique(lay[(r > 1.3e6) & (r < 3.47e6)])[0]
        k = lay == core
        top = F[kind + '_defl'][k][np.argmax(r[k])]
        above = lay == core + 1
        smax = F[kind + '_stress'][above][np.argmin(r[above])]
        lines.append(f'{kind} support: excess eq-minus-polar radius of the level surface at the top of the core '
                     f'{top:+.0f} m; deviatoric stress just above the CMB {smax / 1e6:.1f} MPa '
                     f'(largest in that layer {F[kind + "_stress"][above].max() / 1e6:.1f} MPa); '
                     f'surface topography {float(F[kind + "_surface"]):.0f} m')
    c0, R = read_diurnal()
    V = read_viscous(R)
    lines.append('FCN (sidereal days): CMB m, e_f excess %, formula (A/A_m)(e_f - beta), slipping core stress / '
                 'geometry, viscous core stress / geometry (mean of the extrapolations)')
    for (cmb, ef, b, a), (_, vb, va) in zip(R, V):
        formula = 1 / ((1 / c0) * (E_F_HYD * (1 + ef / 100) - BETA) / (E_F_HYD - BETA))
        lines.append(f'{cmb:5.0f} {ef:6.2f} {formula:7.1f} {b:7.1f} {a:7.1f} {vb:7.1f} {va:7.1f}')
    run.write_text(str(ROOT / 'results' / 'fig_nonhydro_data.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    from provenance import Run
    with Run('figures of the non-hydrostatic Earth (P1 v2 Figs 8-10, S4) and their quoted numbers',
             inputs=['data/PREM.dat', 'data/PREMs.dat', 'results/nonhydro_diurnal.txt',
                     'results/nonhydro_am_sweep_450m.npz']) as run:
        FIG.mkdir(parents=True, exist_ok=True)
        if '--replot' not in sys.argv or not CACHE.exists():
            compute(run)
        summary(run)
        fig_nonhydro_state()
        fig_nonhydro_state('viscous', 'figS_nonhydro_state_viscous')     # Online Resource
        fig_fcn_ef()
        fig_am_sweep()
        for f in ('fig_nonhydro_state', 'figS_nonhydro_state_viscous', 'fig_fcn_ef', 'fig_am_sweep'):
            run.output(str(FIG / f'{f}.pdf'))
