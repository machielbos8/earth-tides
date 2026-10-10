"""Figures for the GJI manuscript.  Run from the project root:

    .venv/bin/python paper/make_figures.py          # compute (cached) + plot
    .venv/bin/python paper/make_figures.py --replot # plot from cache only

Fig. 1  level-surface coordinates (sketch) and Clairaut flattening of PREM
Fig. 2  forces on a displaced parcel (schematic)
Fig. 2  distorted-mesh test: (a) exaggerated level surfaces, (b) error vs K
Fig. 3  convergence of the M2 Love numbers in K and N
Fig. 4  M2 latitude dependence: ellipticity, Coriolis, both
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, BumpGeometry, OMEGA_EARTH, mapping
from earthtides.solver import TidalSolver, ybar
from earthtides.iers import iers_fit

FIG = ROOT / 'paper' / 'figures'
CACHE = ROOT / 'results' / 'figdata.npz'
W_M2 = 1.4053e-4
# validated categorical slots 1-3 (dataviz reference palette); line style is the
# secondary encoding so that the figures survive grayscale printing
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#8a8984'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'legend.frameon': False, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
W1COL, W2COL = 3.3, 6.9          # GJI column / page width (inches)


def compute():
    md = EarthModel(ROOT / 'data/PREMs.dat')
    hard = EarthModel(ROOT / 'data/PREMhard.dat')
    g = ClairautGeometry(md)
    d = {}
    # Fig 1 ----------------------------------------------------------------
    s, eps = [], []
    for i in range(md.n_layers):
        ss = np.linspace(max(md.r0[i], 2e3), md.r1[i], 60)
        s.append(ss)
        eps.append(g.eps(i, ss)[0])
    d['f1_s'], d['f1_eps'] = np.concatenate(s), np.concatenate(eps)
    # Fig 2 ----------------------------------------------------------------
    Ks = np.arange(2, 8)
    for m in (0, 1, 2):
        r0 = TidalSolver(hard, m=m, K=2, N=30, omega_f=0.0).solve()
        err = []
        for K in Ks:
            rb = TidalSolver(hard, BumpGeometry(hard, 0.2), m=m, K=int(K), N=30, omega_f=0.0,
                             background='spherical').solve()
            spur = [abs(v) for k in ('U', 'V', 'W') for n, v in rb[k].items() if n != 2] or [0]
            err.append([abs(rb['h'] - r0['h']), abs(rb['l'] - r0['l']), abs(rb['kk'] - r0['kk']),
                        max(spur)])
        d[f'f2_err{m}'] = np.array(err)
    d['f2_K'] = Ks
    # Fig 3 ----------------------------------------------------------------
    def m2(K, N):
        S = TidalSolver(md, g, m=2, K=int(K), N=int(N), omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=W_M2, omega_rot=OMEGA_EARTH)
        S.solve()
        f = iers_fit(S, g)
        return [f['h0'], f['h2'], f['l2'], f['l1']]
    ref = np.array(m2(7, 40))
    d['f3_ref'] = ref
    d['f3_K'] = np.arange(2, 7)
    d['f3_errK'] = np.array([np.abs(np.array(m2(K, 40)) - ref) for K in d['f3_K']])
    d['f3_N'] = np.array([10, 14, 18, 22, 26, 30, 34])
    d['f3_errN'] = np.array([np.abs(np.array(m2(7, N)) - ref) for N in d['f3_N']])
    # Fig 4 ----------------------------------------------------------------
    th = np.linspace(np.radians(10), np.radians(170), 321)
    d['f4_th'] = th
    for tag, scale, cor in (('ell', 1.0, 0.0), ('cor', 0.0, OMEGA_EARTH), ('both', 1.0, OMEGA_EARTH)):
        gg = ClairautGeometry(md, scale=scale)
        S = TidalSolver(md, gg, m=2, K=5, N=30, omega_f=0.0, coriolis=cor, omega_t=W_M2,
                        omega_rot=OMEGA_EARTH * np.sqrt(scale))
        S.solve()
        a = md.a
        e = gg.eps(md.n_layers - 1, np.array([a]))[0][0]
        norm = (a * (1 + e / 3))**4 / a**3
        ur, ut, up = (v / norm for v in S.surface_field(th))
        phi = np.pi / 2 - th
        Y = ybar(2, 2, th, 1)                       # Schmidt P22 and d/dtheta
        d[f'f4_h_{tag}'] = ur / Y[0]
        d[f'f4_lN_{tag}'] = (-ut) / (-Y[1])          # north = -u_th, d/dphi = -d/dth
        d[f'f4_lE_{tag}'] = up / (-2 * Y[0] / np.cos(phi))
    np.savez(CACHE, **d)
    return dict(np.load(CACHE))


def fig1(d):
    """Inverse flattening of the level surfaces of hydrostatic PREM (column width)."""
    s_all, e_all = d['f1_s'], d['f1_eps']
    fig, ax = plt.subplots(figsize=(W1COL, 2.2))
    ax.plot(s_all / 1e3, 1 / e_all, color=C1)
    for rb, name in ((1221.5, 'ICB'), (3480.0, 'CMB')):
        ax.axvline(rb, color=MUTED, lw=0.5)
        ax.text(rb + 60, 303, name, color=MUTED, fontsize=7)
    ax.set_xlim(0, 6371)
    ax.set_xlabel('mean radius $s$ (km)')
    ax.set_ylabel('inverse flattening $1/\\epsilon$')
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_levels.pdf')


def fig_grid(d):
    """(a) level-surface coordinates with the angular nodes; (b) zoom on one layer."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    s_all, e_all = d['f1_s'], d['f1_eps']
    ex = 40.0                                       # exaggeration of the flattening
    P2 = lambda t: 0.5 * (3 * np.cos(t)**2 - 1)
    Rof = lambda s_km, t: s_km * (1 - 2 / 3 * np.interp(s_km * 1e3, s_all, e_all) * ex * P2(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2COL, 3.3),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))
    th = np.linspace(0, 2 * np.pi, 400)
    interfaces = (1221.5, 3480, 5701, 6371)
    for s_km in interfaces + (2500, 4500, 5300, 6050):
        bold = s_km in interfaces
        R = Rof(s_km, th)
        ax1.plot(R * np.sin(th), R * np.cos(th), color=INK if bold else C1, lw=0.9 if bold else 0.5)
    # colatitude nodes: rays theta = const (Gauss-Legendre nodes, 8 shown)
    xg, _ = np.polynomial.legendre.leggauss(8)
    for t in np.arccos(xg):
        rr = np.linspace(0, 6371, 50)
        R = np.array([Rof(r_, t) for r_ in rr])
        ax1.plot(R * np.sin(t), R * np.cos(t), color=MUTED, lw=0.4, ls=':')
    # point P, r, theta, normal
    t0, s0 = np.radians(50), 4500.0
    R0 = Rof(s0, t0)
    x0, z0 = R0 * np.sin(t0), R0 * np.cos(t0)
    ax1.plot([0, x0], [0, z0], color=INK, lw=0.8)
    ax1.plot([x0], [z0], 'o', color=INK, ms=3)
    ax1.text(x0 - 650, z0 + 250, '$P$', fontsize=8)
    ax1.text(0.6 * x0 + 350, 0.6 * z0 - 700, '$r=R(s,\\theta)$', fontsize=7)
    a_ = np.linspace(0, t0, 30)
    ax1.plot(900 * np.sin(a_), 900 * np.cos(a_), color=INK, lw=0.5)
    ax1.text(330, 1000, '$\\theta$', fontsize=7)
    # zoom region: the upper-mantle layer between s0 and s1, colatitudes tz +- dtz
    s0_, s1_, tz, dtz = 5701.0, 6371.0, np.radians(62), np.radians(7)
    tt = np.linspace(tz - dtz, tz + dtz, 40)
    edge_t = np.concatenate([tt, tt[::-1], tt[:1]])
    edge_s = np.concatenate([np.full(40, s0_), np.full(40, s1_), [s0_]])
    Re = np.array([Rof(a_s, a_t) for a_s, a_t in zip(edge_s, edge_t)])
    ax1.plot(Re * np.sin(edge_t), Re * np.cos(edge_t), color=C2, lw=1.0)
    ax1.text(Rof(s1_, tz) * np.sin(tz) + 150, Rof(s1_, tz) * np.cos(tz) + 150, '(b)', color=C2,
             fontsize=7)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a)', loc='left', fontsize=8)
    # ---- (b): the same layer in (colatitude, radius), nodes along rays
    deg = np.degrees
    for s_km in (s0_, s1_):
        ax2.plot(deg(tt), [Rof(s_km, t) for t in tt], color=INK, lw=1.0)
    for s_km in np.linspace(s0_, s1_, 5)[1:-1]:
        ax2.plot(deg(tt), [Rof(s_km, t) for t in tt], color=C1, lw=0.4)
    Nn = 9
    xk = -np.cos(np.pi * np.arange(Nn) / (Nn - 1))           # Gauss-Lobatto points
    sk = s0_ + (xk + 1) * (s1_ - s0_) / 2
    for t in np.linspace(tz - 0.8 * dtz, tz + 0.8 * dtz, 4):
        r_k = np.array([Rof(x, t) for x in sk])
        ax2.plot([deg(t)] * 2, [r_k[0], r_k[-1]], color=MUTED, lw=0.4, ls=':')
        ax2.plot([deg(t)] * (Nn - 2), r_k[1:-1], 'o', color=C1, ms=3.2)
        ax2.plot([deg(t)] * 2, r_k[[0, -1]], 's', mfc='white', mec=C2, ms=4.5)
    tl = tz + 0.8 * dtz                                        # label the right-most ray
    r_k = np.array([Rof(x, tl) for x in sk])
    xr = deg(tz + dtz) + 0.6
    ax2.annotate('free surface: $\\hat{\\mathbf{n}}\\cdot\\boldsymbol{\\tau}=0$,\n'
                 '$\\phi_1$ matched to the outside', xy=(deg(tl), r_k[-1]),
                 xytext=(xr, r_k[-1] + 40), fontsize=6.5, color=C2, va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.5, color=C2))
    ax2.annotate('interior nodes: equations\n(3)--(4), projected with\neq. (12)',
                 xy=(deg(tl), r_k[4]), xytext=(xr, r_k[4]), fontsize=6.5, color=C1, va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.5, color=C1))
    ax2.annotate('interface: $\\mathbf{u}$, $\\hat{\\mathbf{n}}\\cdot\\boldsymbol{\\tau}$, $\\phi_1$\n'
                 'and gravity flux continuous', xy=(deg(tl), r_k[0]),
                 xytext=(xr, r_k[0] - 40), fontsize=6.5, color=C2, va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.5, color=C2))
    # normal of the upper interface vs the ray, at the left-most ray
    tn = tz - 0.8 * dtz
    dR = (Rof(s1_, tn + 1e-4) - Rof(s1_, tn - 1e-4)) / 2e-4         # dR/dtheta (km/rad)
    Rn = Rof(s1_, tn)
    nr, nt = Rn, -dR                                                # normal ~ (R, -R_theta)
    L = 120.0
    nn = np.hypot(nr, nt)
    ax2.add_patch(FancyArrowPatch((deg(tn), Rn), (deg(tn + L * nt / nn / Rn), Rn + L * nr / nn),
                                  arrowstyle='-|>', mutation_scale=7, color=C2, lw=0.8))
    ax2.text(deg(tn) - 1.6, Rn + 70, '$\\hat{\\mathbf{n}}$', color=C2, fontsize=8)
    ax2.set_xlim(deg(tz - dtz) - 0.5, deg(tz + dtz) + 9.5)
    ax2.set_xlabel('colatitude $\\theta$ (°)')
    ax2.set_ylabel('radius $r$ (km)')
    for sp_ in ('top', 'right'):
        ax2.spines[sp_].set_visible(False)
    ax2.set_title('(b) one layer, $N=9$ Chebyshev nodes per ray', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_grid.pdf')


def fig_parcel(d):
    """Schematic: forces on a displaced parcel. Drawn by parcel_figure() of the manual's figure
    script (one drawing for both), here at column width with the paper's fonts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('manual_figures', ROOT / 'docs' / 'manual' / 'make_figures.py')
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    mf.parcel_figure([FIG / 'fig2_parcel.pdf'], width=W1COL, fs=8)


def fig2(d):
    md = EarthModel(ROOT / 'data/PREMhard.dat')
    bump = BumpGeometry(md, 0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2COL, 2.9),
                                   gridspec_kw=dict(width_ratios=[1, 1.4]))
    th = np.linspace(0, 2 * np.pi, 400)
    for i in range(md.n_layers):
        for s in np.linspace(md.r0[i], md.r1[i], 4)[1:-1]:
            e, de, dde = bump.eps(i, np.array([s]))
            R = mapping(s, th, 6 * e[0], 0, 0)['R'] / 1e3          # exaggerated x6
            ax1.plot(R * np.sin(th), R * np.cos(th), color=C1, lw=0.6)
        R = md.r1[i] / 1e3
        ax1.plot(R * np.sin(th), R * np.cos(th), color=INK, lw=0.8)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a)', loc='left', fontsize=8)
    K = d['f2_K']
    for m, col, ls in ((0, C1, '-'), (1, C2, '--'), (2, C3, ':')):
        e = d[f'f2_err{m}']
        ax2.semilogy(K, np.maximum(e[:, :3].max(1), 1e-16), color=col, ls=ls, marker='o', ms=3,
                     label=f'$m={m}$: max $|\\Delta h|,|\\Delta l|,|\\Delta k|$')
        ax2.semilogy(K, np.maximum(e[:, 3], 1e-16), color=col, ls=ls, marker='s', ms=3, mfc='none',
                     lw=0.8, label=f'$m={m}$: largest spurious coupling')
    ax2.set_xlabel('number of spheroidal degrees $K$')
    ax2.set_ylabel('deviation from the sphere')
    ax2.set_title('(b)', loc='left', fontsize=8)
    ax2.legend(fontsize=6.5, ncol=1, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_distorted_mesh.pdf')


def fig3(d):
    names = ['$h^{(0)}$', '$h^{(2)}$', '$l^{(2)}$', '$l^{(1)}$']
    styles = [(C1, '-'), (C2, '--'), (C3, ':'), (INK, '-.')]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2COL, 2.4), sharey=True)
    for j, (nm, (c, ls)) in enumerate(zip(names, styles)):
        ax1.semilogy(d['f3_K'], np.maximum(np.abs(d['f3_errK'][:, j]), 1e-15), color=c, ls=ls,
                     marker='o', ms=3, label=nm)
        ax2.semilogy(d['f3_N'], np.maximum(np.abs(d['f3_errN'][:, j]), 1e-15), color=c, ls=ls,
                     marker='o', ms=3, label=nm)
    ax1.set_xlabel('number of spheroidal degrees $K$ ($N=40$)')
    ax2.set_xlabel('Chebyshev order per layer $N$ ($K=7$)')
    ax1.set_ylabel('|difference from $K=7$, $N=40$|')
    ax1.set_title('(a)', loc='left', fontsize=8)
    ax2.set_title('(b)', loc='left', fontsize=8)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_convergence.pdf')


def fig4(d):
    phi = 90 - np.degrees(d['f4_th'])
    near_eq = np.abs(phi) < 4                    # l_N divides by dY/dphi = 0 at the equator
    lab = {'ell': 'ellipticity only', 'cor': 'Coriolis only (sphere)', 'both': 'both'}
    sty = {'ell': (C1, '--'), 'cor': (C2, ':'), 'both': (INK, '-')}
    fig, axs = plt.subplots(1, 3, figsize=(W2COL, 2.3), sharex=True)
    for tag in ('ell', 'cor', 'both'):
        c, ls = sty[tag]
        h, lE = d[f'f4_h_{tag}'].real, d[f'f4_lE_{tag}'].real
        lN = np.where(near_eq, np.nan, d[f'f4_lN_{tag}'].real)
        axs[0].plot(phi, 1e4 * (h - h.mean()), color=c, ls=ls, label=lab[tag])
        axs[1].plot(phi, 1e4 * (lN - np.nanmean(lN)), color=c, ls=ls)
        axs[2].plot(phi, 1e4 * (lE - lE.mean()), color=c, ls=ls)
    for ax, t, yl in zip(axs, ('(a) radial', '(b) north', '(c) east'),
                         ('$10^4\\,\\Delta h(\\varphi)$', '$10^4\\,\\Delta l_N(\\varphi)$',
                          '$10^4\\,\\Delta l_E(\\varphi)$')):
        ax.set_title(t, loc='left', fontsize=8)
        ax.set_xlabel('geocentric latitude (°)')
        ax.set_ylabel(yl)
        ax.set_xticks([-60, -30, 0, 30, 60])
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / 'fig4_latitude_M2.pdf')


TOY = ROOT / 'results' / 'figdata_toy.npz'


def compute_toy():
    """Homogeneous spheroid: exact test (no gravity) and gravity-number sweep."""
    import copy
    from earthtides.geometry import ExactSpheroidGeometry
    from earthtides.exact_spheroid import solve, surface_field
    from earthtides.model import G as GRAV
    md0 = EarthModel(ROOT / 'data/HOMOG.dat')
    a, g = md0.a, md0.g_surface
    rho, lam, mu = (float(f(a / 2)) for f in (md0.rho[0], md0.lam[0], md0.mu[0]))
    th = np.linspace(0.05, np.pi - 0.05, 241)

    class Const:
        def __init__(self, f):
            self.f = f

        def eps(self, i, x):
            x = np.asarray(x, float)
            return self.f + 0 * x, 0 * x, 0 * x
    d = {'Ks': np.arange(2, 11), 'fls': np.array([1 / 300, 1 / 30, 1 / 10])}
    for j, fl in enumerate(d['fls']):
        GE = ExactSpheroidGeometry(fl)
        usol = solve(a * GE.A, a * GE.C, lam, mu, rho, g, a, 0)
        urx, utx, _ = surface_field(usol, a * GE.A, a * GE.C, th)
        sc = max(np.abs(urx).max(), np.abs(utx).max())
        for tag, geom in (('ex', GE), ('fo', Const(fl))):
            e = []
            for K in d['Ks']:
                S = TidalSolver(md0, geom, m=0, K=int(K), N=12, omega_f=0.0, gravity=False)
                S.solve()
                ur, ut, _ = S.surface_field(th)
                e.append(max(np.abs(ur - urx).max(), np.abs(ut - utx).max()) / sc)
            d[f'err_{tag}_{j}'] = np.array(e)
    # gravity-number sweep: rigidity scaled so that rho g a / mu takes the values below
    fl = 1 / 300
    Om = np.sqrt(0.8 * fl * GRAV * md0.M / a**3)       # Maclaurin: f = (5/4) m
    ratios = np.geomspace(0.01, 6.0, 22)
    d['ratios'] = ratios
    for grav in (False, True):
        h0, h2 = [], []
        for q in ratios:
            sc = (rho * g * a / q) / mu
            md = copy.copy(md0)
            md.mu = [x * sc for x in md0.mu]
            md.lam = [x * sc for x in md0.lam]
            h0.append(TidalSolver(md, Const(0.0), m=2, K=4, N=16, omega_f=0,
                                  gravity=grav).solve()['h'])
            S = TidalSolver(md, Const(fl), m=2, K=4, N=16, omega_f=0, gravity=grav,
                            omega_rot=Om if grav else 0.0)
            S.solve()
            h2.append(iers_fit(S, Const(fl))['h2'].real / fl)
        d[f'h0_{grav}'], d[f'h2f_{grav}'] = np.array(h0), np.array(h2)
    np.savez(TOY, **d)
    return dict(np.load(TOY))


def fig_toy(t):
    fig, axs = plt.subplots(1, 3, figsize=(W2COL, 2.8))
    ax = axs[0]
    for j, (fl, c) in enumerate(zip(t['fls'], (C1, C2, C3))):
        lab = f'$f=1/{1 / fl:.0f}$'
        ax.semilogy(t['Ks'], np.maximum(t[f'err_ex_{j}'], 1e-16), color=c, marker='o', ms=3)
        ax.semilogy(t['Ks'], t[f'err_fo_{j}'], color=c, ls='--', lw=1.0)
    ax.set_xlabel('number of degrees $K$')
    ax.set_ylabel('error / max $|\\mathbf{u}|$')
    ax.set_title('(a) exact test, no gravity', loc='left', fontsize=8)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=c, label=f'$f=1/{1 / fl:.0f}$') for fl, c in zip(t['fls'], (C1, C2, C3))]
    hl += [Line2D([], [], color=INK, marker='o', ms=3, label='exact shape'),
           Line2D([], [], color=INK, ls='--', lw=1.0, label='first-order shape')]
    ax.legend(handles=hl, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3,
              columnspacing=0.8, handlelength=1.6)
    q = t['ratios']
    for grav, ls, lab in ((False, '--', 'without gravity'), (True, '-', 'with gravity')):
        axs[1].loglog(q, t[f'h0_{grav}'], color=C1, ls=ls, label=lab)
        axs[2].semilogx(q, t[f'h2f_{grav}'] / t[f'h0_{grav}'], color=C1, ls=ls, label=lab)
    axs[1].set_ylabel('$h$ (sphere)')
    axs[2].set_ylabel('$h^{(2)}/(f\\,h)$')
    for ax, tt in zip(axs[1:], ('(b) size and gravity', '(c) relative latitude term')):
        ax.set_xlabel('$\\rho g a/\\mu$')
        ax.axvline(1.96, color=MUTED, lw=0.5)
        ax.set_title(tt, loc='left', fontsize=8)
    axs[1].text(2.15, 0.012, 'Earth', color=MUTED, fontsize=6.5)
    axs[1].legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.33)
    fig.savefig(FIG / 'fig_toy_spheroid.pdf')


def fig_fcn():
    """Diurnal band: h0 and k0 across the FCN resonance, with DDW1999 values."""
    F = dict(np.load(ROOT / 'results' / 'figdata_fcn.npz'))
    x, fcn = F['sweep'], float(F['fcn'])
    fig, axs = plt.subplots(1, 2, figsize=(W2COL, 2.4))
    for ax, key, dkey, lab in ((axs[0], 'h0', 'ddw_h0', '$h^{(0)}$'), (axs[1], 'k0', 'ddw_k0', '$k$')):
        y = F[key].copy()
        for seg in (x < 1.0, (x > 1.0) & (x < fcn), x > fcn):
            ax.plot(x[seg], y[seg], color=C1, label='this paper' if seg is not None and seg[0] else None)
        ax.plot(F['ddw_f'], F[dkey], 'o', mfc='none', mec=C2, ms=5, label='Dehant et al. (1999)')
        ax.axvline(fcn, color=MUTED, lw=0.5, ls='--')
        ax.axvline(1.0, color=MUTED, lw=0.5, ls=':')
        ax.set_xlim(0.985, 1.012)
        ax.set_xlabel('frequency (cycles per sidereal day)')
        ax.set_ylabel(lab)
    axs[0].set_ylim(0.3, 1.2)
    axs[1].set_ylim(0.15, 0.6)
    axs[0].text(fcn + 0.0004, 1.12, 'FCN', color=MUTED, fontsize=7)
    axs[0].text(1.0004, 1.12, 'K$_1$', color=MUTED, fontsize=7)
    axs[0].legend(fontsize=6.5, loc='lower left')
    axs[0].set_title('(a)', loc='left', fontsize=8)
    axs[1].set_title('(b)', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_fcn.pdf')


def fig_budget():
    """Radial amplitude of each choice of Table tab:budget per tidal line, against latitude:
    |dh0 + dh2 P2| |A| W_2m(phi), from the Love-number tables of scripts/displacement_budget.py."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from displacement_budget import variants
    from earthtides.etide import Tamura, W_functions
    cat = Tamura()
    V = variants(cat)
    lines = [('M$_2$', 1.932274, C1, '-'), ('S$_2$', 2.0, C1, '--'), ('N$_2$', 1.895982, C1, ':'),
             ('K$_1$', 1.002738, C2, '-'), ('O$_1$', 0.929536, C2, '--'), ('P$_1$', 0.997262, C2, ':'),
             ('Mf', 0.073202, C3, '-')]
    idx = {}
    for name, f0, c, ls in lines:
        i = np.where(np.abs(np.abs(cat.f_cpd) - f0) < 2e-5)[0]
        idx[name] = i[np.argmax(np.abs(cat.A[i]))]
    lat = np.linspace(0.0, 89.5, 180)
    phi = np.radians(lat)                      # geocentric latitude
    P2 = 1.5 * np.sin(phi)**2 - 0.5
    panels = [('(a) absorption band 1–200 s', 'prem', 'prem200', None),
              ('(b) out-of-phase part of the diurnal resonance', 'mhb', 'mhb_imag_prem', 1),
              ('(c) resonant ocean-loading correction', 'iers', 'mhb', 1),
              ('(d) its in-phase sign', 'iers_flip', 'iers', 1)]
    fig, axs = plt.subplots(2, 2, figsize=(W2COL, 4.4), sharex=True)
    for ax, (title, a, b, only_m) in zip(axs.flat, panels):
        for name, f0, c, ls in lines:
            j = idx[name]
            m = int(cat.m[j])
            if only_m is not None and m != only_m:
                continue
            dh = (V[a].h0[j] - V[b].h0[j]) + (V[a].h2[j] - V[b].h2[j]) * P2
            W, _ = W_functions(2, m, phi)
            ax.plot(lat, 1e3 * np.abs(dh) * abs(cat.A[j]) * np.abs(W), color=c, ls=ls, label=name)
        ax.set_title(title, loc='left', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    for ax in axs[1]:
        ax.set_xlabel('geocentric latitude (°)')
    for ax in axs[:, 0]:
        ax.set_ylabel('radial amplitude (mm)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=7, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / 'fig_budget.pdf')


def fig_stations():
    """Box plots of the hourly radial differences in 2025 per station and source of uncertainty
    (results/station_error_budget.npz from scripts/station_error_budget.py): box 25-75 per cent,
    line the median, whiskers 1-99 per cent."""
    r = np.load(ROOT / 'results' / 'station_error_budget.npz')
    st, kinds, pct = list(r['stations']), list(r['kinds']), r['pct']
    lat = r['lat']
    cols = (C1, C2, C3)
    fig, ax = plt.subplots(figsize=(W2COL, 2.6))
    w = 0.22
    for i in range(len(st)):
        for j in range(len(kinds)):
            x = i + (j - 1) * (w + 0.04)
            p1, q1, med, q3, p99 = pct[i, j]
            ax.add_patch(plt.Rectangle((x - w / 2, q1), w, max(q3 - q1, 1e-3), fc=cols[j],
                                       ec=cols[j], alpha=0.35, lw=0.8))
            ax.plot([x - w / 2, x + w / 2], [med, med], color=cols[j], lw=1.2)
            ax.plot([x, x], [p1, q1], color=cols[j], lw=0.8)
            ax.plot([x, x], [q3, p99], color=cols[j], lw=0.8)
            ax.plot([x - w / 4, x + w / 4], [p1, p1], color=cols[j], lw=0.8)
            ax.plot([x - w / 4, x + w / 4], [p99, p99], color=cols[j], lw=0.8)
    ax.axhline(0, color='0.6', lw=0.5, zorder=0)
    ax.set_xticks(range(len(st)))
    ax.set_xticklabels([f'{s}\n{l:+.0f}°' for s, l in zip(st, lat)])
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylabel('radial difference (mm)')
    labels = ('anelasticity (1–200 s band)', 'ocean model (resonant OTL term)',
              'sign (resonant OTL term)')
    from matplotlib.patches import Patch
    ax.legend([Patch(fc=c, ec=c, alpha=0.5) for c in cols], labels, loc='upper center',
              ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(FIG / 'fig_stations.pdf')


if __name__ == '__main__':
    d = dict(np.load(CACHE)) if ('--replot' in sys.argv and CACHE.exists()) else compute()
    for f in (fig1, fig_grid, fig_parcel, fig2, fig3, fig4):
        f(d)
    t = dict(np.load(TOY)) if ('--replot' in sys.argv and TOY.exists()) else compute_toy()
    fig_toy(t)
    fig_fcn()
    fig_budget()
    fig_stations()
    print('figures written to', FIG)
