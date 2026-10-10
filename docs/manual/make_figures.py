#!/usr/bin/env python3
"""Figures of the manual that are generated from the code itself (so that they cannot drift):

  figures/solver_callgraph.pdf   who calls whom in solver.py, basis.py, jets.py (static analysis with ast),
                                 grouped by role; drawn with Graphviz (dot)
  figures/matrix_structure.png   the system matrix of a PREM M2 solve: non-zeros per block, with
                                 the layers, interface rows and the bordering rows and columns
  figures/cheb_tutorial.png      Chebyshev collocation in one variable: u'' = exp(4x), u(+-1) = 0
  figures/galerkin_tutorial.png  Galerkin projection in one variable: -lap u + (1 + eps P2) u = P2
  figures/jet_tutorial.png       jets and the chain rule in one variable: jets vs finite differences
  figures/ex2_drum.png           example 2: elliptical frame, level curves, nodes, quadrupole solution
  figures/ex3_modela.png         example 3: model a, homogeneous mantle over a liquid core
  figures/ex5_hough.png          example 5: Hough's Earth, rigid flattened shell and liquid core
Usage: cd docs/manual && python make_figures.py
"""
import ast
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1] / 'src' / 'earthtides'
SOURCES = [PKG / 'solver.py', PKG / 'basis.py', PKG / 'jets.py']

# role of every function/method of solver.py (colour, cluster label)
# blocks of the call-graph figure: colour, description of the task, functions
ROLES = {
    'set-up': ('#E8F1FA', 'options, degrees, test functions, liquid layers',
               ['__init__', 'fluid_layers']),
    'basis': ('#EAF5EA', 'trial fields: Chebyshev polynomials in s times Schmidt functions in theta, with derivatives (basis.py)',
              ['cheb_basis', 'ybar', '_trig_derivs', '_schmidt_reduced', '_radial', '_comp', '_fields_at']),
    'physics at the nodes': ('#FDF2E3', 'material, map and reference potential at the nodes; stress and residuals of the equations with jets (jets.py)',
                             ['props', '_stress', 'operator', 'interface', 'normal', 'J']),
    'Galerkin projection': ('#F3E8F7', 'residual onto the kept angular functions (Gauss-Legendre)',
                            ['_project']),
    'assembly': ('#FFFFFF', 'fills the matrix layer by layer: interior, bottom and top rows, free surface',
                 ['solve', '_allocate', '_col', '_interior_rows', '_bottom_rows', '_top_rows', '_surface_rows']),
    'interface rows': ('#FBE9E9', 'liquid-solid conditions, tidal force, Tisserand rows',
                       ['_bnd', '_fs_interface', '_tidal_force_rows', '_tisserand_row']),
    'linear system': ('#F5F5DC', 'scales rows and columns, sparse LU',
                      ['_solve_linear']),
    'output': ('#EFEFEF', 'surface field and Love numbers, rigid rotation removed',
               ['results', 'surface_coefficients', 'surface_field', 'load_love', 'load_potential',
                '_tilt_potential', '_remove_mantle_rotation', '_remove_rigid_rotation', '_rigid_pattern']),
}
EXTERNAL = {'mapping': 'geometry.mapping', 'mapping_from_R': 'geometry.mapping_from_R',
            'chain': 'geometry.chain', 'phys_derivs': 'refpot.phys_derivs',
            'splu': 'scipy splu (sparse LU)', 'solve_dense': 'scipy solve (dense)',
            'ReferencePotential': 'refpot.ReferencePotential'}


def call_graph():
    """Functions and methods of solver.py, basis.py and jets.py and who calls whom (static)."""
    funcs = {}
    for src in SOURCES:
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in ('TidalSolver', 'J'):
                if node.name == 'J':
                    funcs['J'] = None
                    continue
                for f in node.body:
                    if isinstance(f, ast.FunctionDef):
                        funcs[f.name] = f
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                funcs[node.name] = node
    names = set(funcs)
    edges = set()
    for name, f in funcs.items():
        if f is None:
            continue
        for c in ast.walk(f):
            if not isinstance(c, ast.Call):
                continue
            fn = c.func
            callee = None
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'self':
                callee = fn.attr
            elif isinstance(fn, ast.Name):
                callee = fn.id
            elif isinstance(fn, ast.Attribute) and fn.attr in ('splu', 'phys_derivs'):
                callee = fn.attr
            elif isinstance(fn, ast.Attribute) and fn.attr == 'solve' and getattr(fn.value, 'id', '') == 'sla':
                callee = 'solve_dense'
            if callee in names or callee in EXTERNAL:
                if callee != name:
                    edges.add((name, callee))
    return names, edges


def _html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def dot_source(names, edges):
    """Block-level graph: one node per role (name, task, functions), one arrow per pair of blocks
    that call each other (two-headed if both ways), one link per external import."""
    block = {n: r for r, (_, _, ns) in ROLES.items() for n in ns}
    lines = ['digraph solver {', '  rankdir=TB; nodesep=0.35; ranksep=0.45; splines=true; pad=0; margin=0;',
             '  node [fontname="Helvetica"]; edge [color="#444444", arrowsize=0.8, penwidth=1.3];']
    for k, (role, (colour, task, ns)) in enumerate(ROLES.items()):
        funcs = [('J (jet class)' if n == 'J' else n) for n in ns if n in names]
        rows = ', '.join(funcs)
        import textwrap
        fl = '<BR/>'.join(_html(x) for x in textwrap.wrap(rows, 30))
        tl = '<BR/>'.join(_html(x) for x in textwrap.wrap(task, 26))
        label = (f'<<TABLE BORDER="1" CELLBORDER="0" CELLPADDING="4" STYLE="ROUNDED" BGCOLOR="{colour}" COLOR="#777777">'
                 f'<TR><TD><B>{_html(role)}</B></TD></TR><TR><TD><FONT POINT-SIZE="11"><I>{tl}</I></FONT></TD></TR>'
                 f'<TR><TD><FONT FACE="Courier" POINT-SIZE="9">{fl}</FONT></TD></TR></TABLE>>')
        lines.append(f'  "B{k}" [shape=plain, label={label}];')
    bid = {r: f'B{k}' for k, r in enumerate(ROLES)}
    pairs = set()
    ext = set()
    for a, b in edges:
        if a in block and b in block and block[a] != block[b]:
            pairs.add((block[a], block[b]))
        elif a in block and b in EXTERNAL:
            ext.add((block[a], b))
    done = set()
    for a, b in sorted(pairs):
        if (a, b) in done:
            continue
        both = (b, a) in pairs
        done.update({(a, b), (b, a)})
        lines.append(f'  "{bid[a]}" -> "{bid[b]}"' + (' [dir=both];' if both else ';'))
    # layers of the figure: the driver at the top, the foundations at the bottom
    for group in (['assembly', 'linear system'], ['interface rows', 'output'],
                  ['physics at the nodes', 'Galerkin projection'], ['basis', 'set-up']):
        lines.append('  { rank=same; ' + ' '.join(f'"{bid[g]}";' for g in group) + ' }')
    # external imports, one ellipse per module, listing the functions used
    group_of = {'mapping': 'geometry', 'mapping_from_R': 'geometry', 'chain': 'geometry',
                'phys_derivs': 'refpot', 'ReferencePotential': 'refpot',
                'splu': 'scipy', 'solve_dense': 'scipy'}
    shown = {'splu': 'splu (sparse LU)', 'solve_dense': 'solve (dense)', 'phys_derivs': 'phys_derivs',
             'ReferencePotential': 'ReferencePotential', 'mapping': 'mapping',
             'mapping_from_R': 'mapping_from_R', 'chain': 'chain'}
    members = {}
    for _, e in ext:
        members.setdefault(group_of[e], set()).add(e)
    for g, es in sorted(members.items()):
        txt = '<BR/>'.join(_html(shown[e]) for e in sorted(es))
        lines.append(f'  "X{g}" [shape=ellipse, style="filled", fillcolor="#FFFBE0", '
                     f'label=<<B>{g}</B><BR/><FONT POINT-SIZE="9">{txt}</FONT>>, fontsize=11];')
    for r, g in sorted({(r, group_of[e]) for r, e in ext}):
        lines.append(f'  "{bid[r]}" -> "X{g}" [style=dashed, color="#999999"];')
    lines.append('  { rank=same; "Xscipy"; "%s"; }' % bid['interface rows'])
    lines.append('}')
    return '\n'.join(lines)


def callgraph_figure():
    names, edges = call_graph()
    src = dot_source(names, edges)
    (HERE / 'figures' / 'solver_callgraph.dot').write_text(src)
    subprocess.run(['dot', '-Tpdf', '-o', str(HERE / 'figures' / 'solver_callgraph.pdf')],
                   input=src, text=True, check=True)
    missing = sorted(n for n in names if not any(n in ns for _, _, ns in ROLES.values()))
    print('call graph:', len(names), 'functions,', len(edges), 'calls; not in a role:', missing)


def matrix_figure():
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from earthtides import love
    o = love.Options(K=3, N=12)
    S, r = love.solve_wave(o, 2, 1.9322736 * 1.002737909)
    H = np.abs(S.H) > 0
    nL, nb = S.model.n_layers, S.nfld * S.N
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    yy, xx = np.nonzero(H)
    ax.plot(xx, yy, ',', color='#1f3b73', ms=1)
    for i in range(1, nL):
        ax.axhline(i * nb - 0.5, color='#c0504d', lw=0.4)
        ax.axvline(i * nb - 0.5, color='#c0504d', lw=0.4)
    ax.axhline(nL * nb - 0.5, color='#4f9a3a', lw=0.8)
    ax.axvline(nL * nb - 0.5, color='#4f9a3a', lw=0.8)
    ax.set_xlim(-0.5, H.shape[1] - 0.5)
    ax.set_ylim(H.shape[0] - 0.5, -0.5)
    ax.set_xlabel('unknown (Chebyshev coefficient of a field in a layer)')
    ax.set_ylabel('equation (row)')
    ax.set_title(f'PREM, M2, K = 3, N = 12: {H.shape[0]} x {H.shape[1]}, '
                 f'{100 * H.sum() / H.size:.1f} % non-zero', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'matrix_structure.png', dpi=200)
    print('matrix:', H.shape, f'{100 * H.sum() / H.size:.2f} % non-zero, {nL} layers, block {nb}')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# Okabe-Ito colour-blind-safe palette (as in the Hector manual and paper)
OI_BLUE, OI_ORANGE, OI_GREEN, OI_PURPLE = '#0072B2', '#E69F00', '#009E73', '#CC79A7'
OI_VERMILLION, OI_SKY, INK, DARK = '#D55E00', '#56B4E9', '#000000', '#333333'


def parcel_figure(paths=None, width=3.3, fs=8, png=None):
    """The four forces on a displaced parcel (manual, 'The equations of a deforming Earth'; paper 1,
    Fig. 2, drawn by paper/make_figures.py with this function and the paper's fonts): reference
    state with gravity g = -grad Phi0, p0 and rho0 increasing with depth; the parcel displaced by u,
    sheared and compressed. Surface forces on the faces: elastic stress, normal and shear (1), higher
    reference pressure on all sides (2). Body forces from the centre: weight of the density change
    (3), gravity of the moved mass and of the tide (4). Okabe-Ito colours plus line styles."""
    plt = _plt()
    from matplotlib.patches import Polygon, FancyArrowPatch
    paths = paths or [HERE / 'figures' / 'parcel.pdf']
    X0, X1, ys = 0.3, 8.6, [0.4, 1.6, 2.8, 4.0, 5.2, 6.4]
    fig = plt.figure(figsize=(width, width * 1.0))
    ax = fig.add_axes([0.02, 0.27, 0.96, 0.72])
    def arrow(p, q, c=INK, lw=1.2, ls='-', ms=8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw,
                                     linestyle=ls, shrinkA=0, shrinkB=0))
    def txt(x, y, t, **kw):
        ax.text(x, y, t, fontsize=kw.pop('fontsize', fs), color=kw.pop('color', INK), **kw)
    # reference state: level surfaces, pressure and density increase downward (darker)
    for k in range(5):
        ax.fill_between([X0, X1], ys[k], ys[k + 1], color='0.5', alpha=0.05 + 0.05 * (4 - k), lw=0)
    for y in ys:
        ax.plot([X0, X1], [y, y], color=DARK, lw=0.5, ls=':')
    txt(X1, ys[-1] + 0.08, r'level surfaces: $p_0$, $\rho_0$, $\Phi_0$ constant', color=DARK,
        ha='right', va='bottom')
    txt(X1, ys[0] - 0.08, r'deeper: larger $p_0$ and $\rho_0$', color=DARK, ha='right', va='top')
    arrow((0.65, 6.05), (0.65, 4.45), DARK, lw=1.3)
    txt(0.8, 5.25, '$\\mathbf{g}=-\\nabla\\Phi_0$\ngravity', color=DARK, va='center')
    # before: reference parcel
    ref = np.array([[2.95, 4.45], [4.4, 4.45], [4.4, 5.9], [2.95, 5.9]])
    ax.add_patch(Polygon(ref, closed=True, fill=False, ec=INK, lw=1.2))
    txt(3.65, 6.0, r'before, at $\mathbf{x}$', ha='center', va='bottom')
    # displacement and after: sheared and smaller (compressed, denser)
    arrow((4.5, 4.45), (5.0, 3.45), INK, lw=1.4)
    txt(5.05, 4.2, r'$\mathbf{u}$', fontsize=fs + 2)
    dfm = np.array([[5.0, 1.75], [6.5, 1.85], [6.8, 3.1], [5.3, 3.0]])
    ax.add_patch(Polygon(dfm, closed=True, fill=True, fc=OI_SKY, alpha=0.35, ec=OI_BLUE, lw=1.2,
                         ls='--'))
    c = dfm.mean(axis=0)
    txt(4.15, 1.95, 'after, at $\\mathbf{x}+\\mathbf{u}$:\nsmaller, so denser', ha='right',
        va='center')
    # (1) surface force, elastic stress: normal across the faces, shear along them
    arrow((6.75, 2.5), (7.4, 2.5), OI_VERMILLION)
    arrow((5.12, 2.4), (4.47, 2.4), OI_VERMILLION)
    arrow((5.4, 3.17), (5.85, 3.2), OI_VERMILLION, lw=1.0, ms=7)
    arrow((5.75, 1.67), (5.25, 1.64), OI_VERMILLION, lw=1.0, ms=7)
    txt(7.48, 2.5, '(1)', va='center')
    txt(5.6, 3.27, '(1)', ha='center', va='bottom')
    # (2) surface force, higher reference pressure of the new level, on all sides
    for (x, y), (dx, dy) in zip(dfm, ((-1, -1), (1, -1), (1, 1), (-1, 1))):
        arrow((x + 0.38 * dx, y + 0.38 * dy), (x + 0.06 * dx, y + 0.06 * dy), OI_ORANGE, lw=1.3,
              ls='--', ms=7)
    txt(dfm[2, 0] + 0.42, dfm[2, 1] + 0.42, '(2)', va='bottom')
    # body forces from the centre: (3) weight of the density change, (4) phi1 + phiT
    ax.plot(*c, 'o', color=INK, ms=3)
    arrow(tuple(c), (c[0], 1.0), OI_GREEN, lw=1.6, ls='-.')
    txt(c[0] + 0.18, 1.15, '(3)', va='center')
    tip = (c[0] + 0.6, c[1] + 2.45)
    arrow(tuple(c), tip, OI_PURPLE, lw=1.6, ls=(0, (1, 1.5)))
    txt(tip[0] + 0.12, tip[1] + 0.05, '(4) $\\phi_1+\\phi_T$', ha='left', va='center')
    ax.set_xlim(X0 - 0.05, X1 + 0.05)
    ax.set_ylim(0.0, 7.0)
    ax.set_aspect('equal')
    ax.axis('off')
    # legend, left-aligned with the layer box
    ax.apply_aspect()
    pos = ax.get_position()
    x_left = pos.x0 + pos.width * (X0 - (X0 - 0.05)) / (X1 - X0 + 0.1)
    key = [(OI_VERMILLION, '-', r'(1) surface: elastic stress, normal and shear, $\nabla\!\cdot\boldsymbol{\tau}$'),
           (OI_ORANGE, '--', r'(2) surface: pressure on all sides, $-\nabla(\rho_0\,\mathbf{u}\cdot\nabla\Phi_0)$'),
           (OI_GREEN, '-.', r'(3) body: weight of density change, $\nabla\!\cdot(\rho_0\mathbf{u})\,\nabla\Phi_0$'),
           (OI_PURPLE, (0, (1, 1.5)), r'(4) body: moved mass and tide, $-\rho_0\nabla(\phi_1+\phi_T)$')]
    lax = fig.add_axes([x_left, pos.y0 - 0.215, pos.x1 - x_left, 0.2])
    lax.axis('off')
    for k, (col, ls, t) in enumerate(key):
        y = 0.9 - 0.26 * k
        lax.plot([0.0, 0.07], [y, y], color=col, lw=1.6, ls=ls)
        lax.text(0.09, y, t, fontsize=fs - 0.5, va='center')
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    for path in paths:
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(png, dpi=200, bbox_inches='tight', pad_inches=0.02)


def onsala_figure():
    """etide at Onsala (57.395 N, 11.926 E), 30 days hourly from 2026-01-01, as
    `etide 57.395 11.926 --start 2026-01-01 --days 30 --step 3600`: up, north, east with the IERS
    Love numbers, and the difference made by the Love numbers of anelastic PREM (--love prem)."""
    import sys
    sys.path.insert(0, str(PKG.parent))
    from earthtides.etide import displacement
    plt = _plt()
    mjd = 61041.0 + np.arange(720) / 24.0
    iers = displacement(mjd, 57.395, 11.926, love='iers')
    prem = displacement(mjd, 57.395, 11.926, love='prem')
    t = mjd - 61041.0
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True)
    for ax, u, name, col in zip(axs[:3], iers, ('up', 'north', 'east'),
                                (OI_BLUE, OI_ORANGE, OI_GREEN)):
        ax.plot(t, 100 * u, color=col, lw=0.8)
        ax.set_ylabel(f'{name} [cm]')
        ax.axhline(0, color=DARK, lw=0.4)
        ax.grid(True, lw=0.3)
    for u0, u1, name, col, ls in zip(iers, prem, ('up', 'north', 'east'),
                                     (OI_BLUE, OI_ORANGE, OI_GREEN), ('-', '--', ':')):
        axs[3].plot(t, 1000 * (u1 - u0), color=col, lw=0.9, ls=ls, label=name)
    axs[3].set_ylabel('PREM $-$ IERS [mm]')
    axs[3].legend(fontsize=8, frameon=False, ncol=3, loc='upper right')
    axs[3].grid(True, lw=0.3)
    axs[3].set_xlabel('days since 2026-01-01 00:00 UTC')
    axs[3].set_xlim(0, 30)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'onsala.png', dpi=200)
    d = 1000 * (np.array(prem) - np.array(iers))
    print('Onsala: up', 100 * iers[0].min(), 100 * iers[0].max(), 'cm; PREM-IERS max |d| mm',
          np.abs(d).max(axis=1), 'rms', np.sqrt((d**2).mean(axis=1)))


def cheb_tutorial_solve(N):
    """u'' = exp(4x) on [-1, 1], u(+-1) = 0 by Chebyshev collocation (Trefethen 2000, Program 13):
    N - 2 interior Gauss-Lobatto rows of the equation, two boundary rows. Returns the amplitudes
    a_k, the matrix, the right-hand side and the interior nodes."""
    from numpy.polynomial import chebyshev as C
    xj = np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
    eye = np.eye(N)
    M, b = np.zeros((N, N)), np.zeros(N)
    M[:N - 2] = C.chebval(xj, C.chebder(eye, 2)).T          # T_k''(x_j)
    b[:N - 2] = np.exp(4 * xj)
    M[N - 2] = C.chebval(1.0, eye)                           # u(1) = 0
    M[N - 1] = C.chebval(-1.0, eye)                          # u(-1) = 0
    return np.linalg.solve(M, b), M, b, xj


def cheb_tutorial_figure():
    from numpy.polynomial import chebyshev as C
    plt = _plt()
    exact = lambda x: (np.exp(4 * x) - x * np.sinh(4) - np.cosh(4)) / 16
    xx = np.linspace(-1, 1, 801)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax1.plot(xx, exact(xx), 'k', lw=2.5, label='exact')
    for N, col in ((6, '#0072B2'), (8, '#D55E00')):
        a, _, _, xj = cheb_tutorial_solve(N)
        ax1.plot(xx, C.chebval(xx, a), color=col, lw=1.2, label=f'N = {N}')
        ax1.plot(xj, C.chebval(xj, a), 'o', color=col, ms=4)
        ax1.plot([-1, 1], [0, 0], 'o', mfc='white', color=col, ms=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title('(a) solution; dots: collocation nodes,\nopen circles: boundary conditions', fontsize=9)
    Ns = np.arange(4, 25, 2)
    err = [np.abs(C.chebval(xx, cheb_tutorial_solve(N)[0]) - exact(xx)).max() for N in Ns]
    ax2.semilogy(Ns, err, 'o-', color='#0072B2', ms=4)
    ax2.set_xlabel('number of Chebyshev polynomials N')
    ax2.set_ylabel('maximum error')
    ax2.set_title('(b) exponential convergence', fontsize=9)
    ax2.grid(True, which='major', lw=0.3)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'cheb_tutorial.png', dpi=200)
    print('Chebyshev tutorial: max errors', dict(zip(Ns.tolist(), [f'{e:.1e}' for e in err])))


def jet_tutorial_E(r, h=None):
    """Radial equation E = d tau_rr/dr + 2 (tau_rr - tau_tt)/r of an elastic sphere in radial motion,
    u = r^2, lambda = 2 + r, mu = 1 + r^2, with the fields stored on the level coordinate s of the
    map r = s (1 + 0.1 s^2): chain rule for u_r, u_rr, then jets (h=None) or a central difference
    of tau_rr with step h. Exact: E = 16 + 8 r + 16 r^2."""
    import sys
    sys.path.insert(0, str(PKG.parent))
    from earthtides.jets import J
    s = np.atleast_1d(r).astype(float)
    for _ in range(60):                              # invert the map by Newton
        s = s - (s * (1 + 0.1 * s**2) - r) / (1 + 0.3 * s**2)
    R, R1, R2 = s * (1 + 0.1 * s**2), 1 + 0.3 * s**2, 0.6 * s
    u_s, u_ss = 2 * R * R1, 2 * R1**2 + 2 * R * R2       # u(s) = R(s)^2
    s_r, s_rr = 1 / R1, -R2 / R1**3
    u_r, u_rr = u_s * s_r, u_ss * s_r**2 + u_s * s_rr
    if h is None:
        Rj, u, ur = J(R, 1.0), J(R**2, u_r), J(u_r, u_rr)
        lam, mu = J(2 + R, 1.0), J(1 + R**2, 2 * R)
        D = ur + 2 * u / Rj
        trr, ttt = lam * D + 2 * mu * ur, lam * D + 2 * mu * u / Rj
        return trr.r + 2 * (trr.v - ttt.v) / R
    tau = lambda x: (2 + x) * 4 * x + 2 * (1 + x**2) * 2 * x, lambda x: (2 + x) * 4 * x + 2 * (1 + x**2) * x
    return (tau[0](R + h) - tau[0](R - h)) / (2 * h) + 2 * (tau[0](R) - tau[1](R)) / R


def jet_tutorial_figure():
    plt = _plt()
    exact = lambda r: 16 + 8 * r + 16 * r**2
    r0 = 0.75 * (1 + 0.1 * 0.75**2)
    hs = np.logspace(-9, -1, 33)
    efd = [abs(jet_tutorial_E(r0, h)[0] - exact(r0)) / exact(r0) for h in hs]
    ejet = max(abs(jet_tutorial_E(r0)[0] - exact(r0)) / exact(r0), 1e-16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax1.loglog(hs, efd, 'o-', color='#D55E00', ms=3, label='finite difference')
    ax1.axhline(ejet, color='#0072B2', lw=1.5, label='jets')
    ax1.set_xlabel('step h')
    ax1.set_ylabel('relative error of E')
    ax1.set_ylim(1e-17, 1e-1)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(f'(a) at r = {r0:.4f}', fontsize=9)
    ax1.grid(True, which='major', lw=0.3)
    rr = np.linspace(0.3, 1.0, 141)
    ax2.semilogy(rr, np.maximum(np.abs(jet_tutorial_E(rr) - exact(rr)) / exact(rr), 1e-16), color='#0072B2', lw=1.2, label='jets')
    ax2.semilogy(rr, np.abs(jet_tutorial_E(rr, 1e-5) - exact(rr)) / exact(rr), color='#D55E00', lw=1.2, label='finite difference, h = 1e-5')
    ax2.set_xlabel('r')
    ax2.set_ylabel('relative error of E')
    ax2.set_ylim(1e-17, 1e-1)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title('(b) along the radius', fontsize=9)
    ax2.grid(True, which='major', lw=0.3)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'jet_tutorial.png', dpi=200)
    print('jet tutorial: E jets', jet_tutorial_E(r0)[0], 'exact', exact(r0), 'best FD', min(efd))


GAL_EPS = 0.3


def galerkin_tutorial_solve(degs, eps=GAL_EPS):
    """-lap_S u + (1 + eps P2(mu)) u = P2(mu) on the sphere (axisymmetric, mu = cos theta): trial
    u = sum a_n P_n over the kept degrees, residual projected on every kept P_k by Gauss-Legendre
    quadrature (2 n_max + 24 points, exact). Returns the amplitudes, the matrix, the right-hand side."""
    from numpy.polynomial import legendre as L
    P = lambda n, x: L.legval(x, np.eye(n + 1)[n])
    K = len(degs)
    mu, w = L.leggauss(2 * max(degs) + 24)
    A, b = np.zeros((K, K)), np.zeros(K)
    for i, k in enumerate(degs):
        for j, n in enumerate(degs):
            A[i, j] = np.sum(w * (n * (n + 1) + 1 + eps * P(2, mu)) * P(n, mu) * P(k, mu))
        b[i] = np.sum(w * P(2, mu) * P(k, mu))
    return np.linalg.solve(A, b), A, b


def galerkin_tutorial_figure():
    from numpy.polynomial import legendre as L
    plt = _plt()
    P = lambda n, x: L.legval(x, np.eye(n + 1)[n])
    th = np.linspace(0, np.pi, 721)
    mu = np.cos(th)
    u = lambda a, degs, x: sum(a[i] * P(n, x) for i, n in enumerate(degs))
    sets = {1: [2], 2: [0, 2], 3: [0, 2, 4], 4: [0, 2, 4, 6], 5: [0, 2, 4, 6, 8], 6: [0, 2, 4, 6, 8, 10]}
    dr = list(range(0, 40, 2))
    ref = galerkin_tutorial_solve(dr)[0]
    uref = u(ref, dr, mu)
    sc = np.abs(uref).max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for K, col in ((1, '#0072B2'), (2, '#D55E00'), (3, '#009E73')):
        degs = sets[K]
        a = galerkin_tutorial_solve(degs)[0]
        # residual E = -lap u + (1 + eps P2) u - P2
        E = sum(a[i] * (n * (n + 1) + 1 + GAL_EPS * P(2, mu)) * P(n, mu) for i, n in enumerate(degs)) - P(2, mu)
        ax1.plot(90 - np.degrees(th), E, color=col, lw=1.3, label=f'degrees {", ".join(map(str, degs))}')
    ax1.set_xlabel('latitude (degrees)')
    ax1.set_ylabel('residual E')
    ax1.set_xlim(-90, 90)
    ax1.legend(fontsize=7.5, frameon=False)
    ax1.set_title('(a) residual: contains none of the kept degrees', fontsize=9)
    Ks = sorted(sets)
    err = [np.abs(u(galerkin_tutorial_solve(sets[K])[0], sets[K], mu) - uref).max() / sc for K in Ks]
    a0 = galerkin_tutorial_solve(dr, 0.0)[0]
    ap = galerkin_tutorial_solve(dr, 1e-6)[0]
    u1 = u(a0, dr, mu) + GAL_EPS * (u(ap, dr, mu) - u(a0, dr, mu)) / 1e-6
    e1 = np.abs(u1 - uref).max() / sc
    ax2.semilogy(Ks, err, 'o-', color='#0072B2', ms=4, label='Galerkin, K degrees')
    ax2.axhline(e1, color='#D55E00', ls='--', lw=1.0, label='first-order perturbation')
    ax2.set_xlabel('number of kept degrees K')
    ax2.set_ylabel('maximum relative error')
    ax2.set_title('(b) exponential convergence', fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'galerkin_tutorial.png', dpi=200)
    print('Galerkin tutorial: rel. errors', dict(zip(Ks, [f'{e:.1e}' for e in err])), f'first order {e1:.1e}')


def drum_figure():
    """Example 2: level curves r = s R(theta) of an ellipse (f = 0.2, to be visible), Chebyshev
    nodes on a few rays, and the exact solution for the quadrupole load."""
    import importlib.util
    plt = _plt()
    spec = importlib.util.spec_from_file_location('drum', HERE.parents[1] / 'examples' / 'ex2' / 'drum.py')
    drum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drum)
    f = 0.2
    a, b = 1.0, 1.0 - f
    th = np.linspace(0, 2 * np.pi, 400)
    R = 1 / np.sqrt(np.cos(th)**2 / a**2 + np.sin(th)**2 / b**2)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    X, Y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
    inside = X**2 / a**2 + Y**2 / b**2 <= 1
    U = np.where(inside, drum.exact('quadrupole', a, b, X, Y), np.nan)
    pc = ax.pcolormesh(X, Y, U, cmap='RdBu_r', shading='auto',
                       vmin=-np.nanmax(np.abs(U)), vmax=np.nanmax(np.abs(U)))
    for s in np.linspace(0.2, 1.0, 5):
        ax.plot(s * R * np.cos(th), s * R * np.sin(th), color='k', lw=1.4 if s == 1 else 0.6)
    xn = (np.cos(np.pi * np.arange(13) / 12) + 1) / 2
    for t in np.radians([0, 30, 60, 90]):
        Rt = 1 / np.sqrt(np.cos(t)**2 / a**2 + np.sin(t)**2 / b**2)
        ax.plot(xn * Rt * np.cos(t), xn * Rt * np.sin(t), 'o', ms=2.5, color='#333333')
        ax.plot([0, Rt * np.cos(t)], [0, Rt * np.sin(t)], ':', color='#333333', lw=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.9, 0.9)
    ax.set_axis_off()
    fig.colorbar(pc, ax=ax, shrink=0.7, label='deflection u (quadrupole load x$^2$ - y$^2$)')
    ax.set_title('Elliptical drum (flattening 0.2): level curves and nodes', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'ex2_drum.png', dpi=200)


def _cross_section(ax, layers, flat=0.0, labels=(), axis=None):
    """Draw a meridional cross-section: layers = [(radius, colour), ...] from outside in."""
    from matplotlib.patches import Ellipse
    for r, colour in layers:
        ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r * (1 - flat), facecolor=colour, edgecolor='k', lw=1.0))
    for x, y, txt in labels:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()


def modela_figure():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    _cross_section(ax, [(1.0, '#C9A97A'), (0.55, '#7FB3D5')],
                   labels=[(0, 0.77, 'mantle (solid)\n' r'$\rho$ = 4460 kg m$^{-3}$' '\n'
                            r'$\mu$ = 174 GPa, $\lambda$ = 231 GPa'),
                           (0, 0.0, 'liquid core\n' r'$\rho$ = 11020 kg m$^{-3}$' '\n'
                            r'$\mu$ = 0, $\lambda$ = 950 GPa'),
                           (0, -0.7, 'r$_c$ = 0.55 a, slip at the boundary')])
    ax.set_title('Example 3, model a', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'ex3_modela.png', dpi=200)


def hough_figure():
    from matplotlib.patches import FancyArrowPatch
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    flat = 0.12                                       # exaggerated (Hough's Earth: 1/231)
    _cross_section(ax, [(1.0, '#B0B0B0'), (3480 / 6371, '#7FB3D5')], flat=flat,
                   labels=[(0.0, 0.0, 'liquid core\nhomogeneous,\nincompressible'),
                           (0.78, -0.62, 'rigid\nmantle'),
                           (0, -1.08, 'similar Maclaurin spheroids (flattening exaggerated)')])
    for ang, col, lab, ha, dx in ((0.0, 'k', r'$\Omega$ (mantle)', 'right', -0.04),
                                  (np.radians(14), '#1F4E79', 'core axis', 'left', 0.04)):
        x, y = 1.05 * np.sin(ang), 1.05 * np.cos(ang)
        ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle='-|>', mutation_scale=10, color=col, lw=1.2))
        ax.text(x + dx, y, lab, fontsize=8.5, color=col, va='bottom', ha=ha)
    ax.set_title('Example 5: the free core nutation of Hough (1895)', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'ex5_hough.png', dpi=200)


if __name__ == '__main__':
    cheb_tutorial_figure()
    galerkin_tutorial_figure()
    jet_tutorial_figure()
    parcel_figure()
    onsala_figure()
    callgraph_figure()
    matrix_figure()
    drum_figure()
    modela_figure()
    hough_figure()
