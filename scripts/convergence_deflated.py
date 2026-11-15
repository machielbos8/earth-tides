# -*- coding: utf-8 -*-
#
# Convergence test (2026-10-17, lab notes item 10): the scaled matrix of a run
# with a rotating fluid core and a solid inner core has two near-null right
# singular vectors (sigma 1e-16 - 1e-12), confined to the fluid core and its
# highest-degree U, V fields. The solution component along them is set by
# round-off and leaks ~1e-5 - 1e-3 into the surface Love numbers. Here they
# are removed (truncated SVD): right singular vectors from block inverse
# iteration with the sparse LU; removed if sigma < 1e-10 AND > 99 % of the
# energy is in the fluid layers (keeps the physical W1 rotational modes of
# mantle and inner core, sigma 3e-11 and 3e-9 near the diurnal resonances).
# Scratch subclass; production code untouched. ->
# results/convergence_deflated.txt
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Convergence test (2026-10-17, lab notes item 10): the scaled matrix of a run with a rotating fluid
core and a solid inner core has two near-null right singular vectors (sigma 1e-16 - 1e-12), confined
to the fluid core and its highest-degree U, V fields. The solution component along them is set by
round-off and leaks ~1e-5 - 1e-3 into the surface Love numbers. Here they are removed (truncated SVD):
right singular vectors from block inverse iteration with the sparse LU; removed if sigma < 1e-10 AND
> 99 % of the energy is in the fluid layers (keeps the physical W1 rotational modes of mantle and
inner core, sigma 3e-11 and 3e-9 near the diurnal resonances). Scratch subclass; production code
untouched. -> results/convergence_deflated.txt"""
import sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, scipy.linalg as sla
from multiprocessing import Pool
from earthtides import solver as sv
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.iers import iers_fit

INFO = {}
def defl_solve(self, H, f):
    Hc = H.tocsr() if hasattr(H, 'tocsr') else sps.csr_matrix(H)     # hybrid matrix since solver-6
    rs = abs(Hc).max(axis=1).toarray().ravel(); Hc = sps.diags(1 / rs) @ Hc
    cs = abs(Hc).max(axis=0).toarray().ravel(); cs[cs == 0] = 1
    A = (Hc @ sps.diags(1 / cs)).tocsc()
    lu = spla.splu(A, permc_spec='COLAMD')
    b = f / rs
    x = lu.solve(b)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((A.shape[0], 6))
    for _ in range(4):
        Y = np.column_stack([lu.solve(lu.solve(y, trans='T')) for y in Y.T])
        Y, _ = np.linalg.qr(Y)
    s = np.linalg.svd(A @ Y, compute_uv=False)
    # Ritz: singular values of A on the subspace
    Uq, sq, Vq = np.linalg.svd(A @ Y, full_matrices=False)
    V = Y @ Vq.T                     # approx right singular vectors, sigma sq
    fl = self.fluid_layers(); nb = self._nb
    mask = np.zeros(A.shape[0], bool)
    for i, isf in enumerate(fl):
        if isf:
            mask[i * nb:(i + 1) * nb] = True
    keep = (sq < 1e-10) & ((V[mask]**2).sum(0) > 0.99)
    INFO['n'] = int(keep.sum())
    INFO['s'] = np.sort(sq)
    Vn = V[:, keep]
    xc = x - Vn @ (Vn.T @ x)
    return xc / cs

class Defl(sv.TidalSolver):
    _solve_linear = defl_solve

def run(a):
    m, sig, K, N = a
    md = EarthModel('data/PREM.dat'); g = ClairautGeometry(md)
    S = Defl(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
             omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
             frame='mantle')
    S.solve(); fx = iers_fit(S, g)
    return (m, sig, K, N) + tuple(fx[k].real for k in ('h0', 'h2', 'l2', 'l1')) + (INFO['n'], str(np.array2string(INFO['s'], precision=1)))

if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('inviscid core: spread between solvers after removing the near-null space (S9)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    res_ = [(3, 24), (4, 24), (4, 30), (5, 30), (6, 30), (7, 40)]
    waves = [(2, 1.9225), (2, 1.92700), (2, 1.9475), (1, 0.92700), (1, 0.905)]
    jobs = [(m, s, K, N) for m, s in waves for K, N in res_]
    with Pool(4) as p:          # Mac memory budget (2026-11-10)
        res = p.map(run, jobs, chunksize=1)
    with open('results/convergence_deflated.txt', 'w') as fo:
        fo.write('m sigma K N h0 h2 l2 l1 n_removed smallest_sv\n')
        for r in res:
            fo.write('%d %.5f %d %d %.10f %.10f %.10f %.10f %d %s\n' % r)
    for m, s in waves:
        rr = np.array([r[4:8] for r in res if r[1] == s])
        line = ('m=%d %.5f spread(1e-5) h0 %.4f h2 %.4f l2 %.4f l1 %.4f' % ((m, s) + tuple(1e5 * (rr.max(0) - rr.min(0)))))
        print(line)
        open('results/convergence_deflated.txt', 'a').write('# ' + line + '\n')
    _run.output('results/convergence_deflated.txt')
    _run.__exit__(None, None, None)
