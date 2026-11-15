# -*- coding: utf-8 -*-
#
# FCN period of the consistent non-hydrostatic Earth (Table 4, route B) with a viscous core:
# complex pole of k at finite Ekman-layer thickness delta, and the limit delta -> 0.
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""FCN period of the consistent non-hydrostatic Earth (Table 4, route B) with a viscous core:
complex pole of k at finite Ekman-layer thickness delta, and the limit delta -> 0.

Machiel (2026-10-18): the 431 days must not rest on an inviscid core. Same state as
scripts/nonhydro_diurnal.py (elastic PREM, excess J2 1.14e-5, excess CMB flattening CMB m,
FlatSupport / FlatState / FlatMap), but EarthModel.viscous_core (no slip, Ekman sub-layers). The
viscous model splits the core into sub-layers; the state and the map address layers by index,
so _ByParent maps every sub-layer to its parent layer (the sub-layers carry the parent's
polynomials, re-expanded, and the state is evaluated at physical radii). Welded boundaries: the
slip terms of route B (slipping fluid-solid boundaries only) drop out, the P1(tau).n tractions stay.
FCN: k at 4 frequencies around the resonance, k = (b x - d) / (x - c) fitted (complex): c = complex
FCN frequency (cpsd); period = 1 / (Re c - 1) sidereal days, damping Im c.
Usage: python scripts/nonhydro_fcn_viscous.py CMB_M K N DELTA_KM [DELTA_KM ...]  (env ROUTE, CORE_STRESS, CORE_DP,
       ANELASTIC=1: absorption band in the tidal run, the state stays elastic)
       -> results/nonhydro_fcn_viscous_<case>_K<K>_N<N>.txt
CMB_M = -1: hydrostatic PREM (control: earlier 423.0 / 438.6 / 447.9 sd at 2 / 1 / 0.5 km, K 6).
"""
import os
import sys
import time
import warnings
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
sys.path.insert(0, 'scripts')
from provenance import Run

SHELLS = [(3480e3, 5701e3), (5701e3, 6346.6e3)]
SIGMA = (1.0012, 1.0016, 1.0030, 1.0036)
CORE_STRESS = os.environ.get('CORE_STRESS', '1') == '1'    # static -dp I in the welded core (default)
ROUTE = os.environ.get('ROUTE', 'B')                        # 'A': geometry + density, hydrostatic stress
ANELASTIC = os.environ.get('ANELASTIC', '0') == '1'           # tidal run with PREM's absorption band (state elastic)
CORE_DP = os.environ.get('CORE_DP', 'static')               # core stress: 'static' (default) | 'gauge' (old)


class _RouteA:
    """Route A of nonhydro_diurnal.py: the state's density, no deviatoric stress, no slip terms."""

    def __init__(self, st):
        self.st = st

    def drho(self, *a):
        return self.st.drho(*a)

    def tau(self, *a):
        return None


class _ByParent:
    """Wraps a state or a geometry built on the unsplit model: layer-indexed calls of the split
    (viscous) model go to the parent layer; everything else passes through."""

    INDEXED = ('drho', 'tau', 'map_derivs', 'eps')

    def __init__(self, obj, parent):
        self._obj, self._parent = obj, parent

    def __getattr__(self, name):
        f = getattr(self._obj, name)
        if name in self.INDEXED:
            return lambda i, *a, **k: f(self._parent[i], *a, **k)
        if name == 'slip':                       # (ifl, rb, th): ifl is a layer index too
            return lambda i, *a, **k: f(self._parent[i], *a, **k)
        if name == 'base':                       # FlatMap.base, the background map (solver._normal_bg)
            return _ByParent(f, self._parent)
        return f


class _CoreStress(_ByParent):
    """As _ByParent, and in the (welded, viscous) core the static stress -dp I relative to the
    hydrostatic pressure: dp = p0'(s) s_r q(s, theta) is the pressure the core's non-isobaric map
    (gauge q) carries, which route B passes to a SLIPPING boundary through slip(); a welded boundary
    needs it as the fluid's own initial stress (C(u) = 0 for an isotropic tau, so only the
    interface traction P1(tau).n changes)."""

    def _dp(self, s, t):
        st = self._obj
        if CORE_DP == 'static':                  # 2026-10-31: from the static solve (balances tau_nn)
            return st.core_dp(s, t)
        p1, _ = st._p0(st.ioc, s)                # 'gauge': the original form, degree 0 off (task C)
        return p1[:, None] * st.fields(st.ioc, s, t)[0]['map']['s_r'] * st._q(s, t)

    def __getattr__(self, name):
        if name != 'tau':
            return _ByParent.__getattr__(self, name)
        st, par = self._obj, self._parent

        def tau(i, s2, th, mp):
            if par[i] != st.ioc:
                return st.tau(par[i], s2, th, mp)
            from earthtides.jets import J
            from earthtides.nonhydro import _sderivs
            s = np.asarray(s2, float).ravel()
            t = np.asarray(th, float).ravel()
            r0, r1 = st.md.r0[st.ioc], st.md.r1[st.ioc]
            f0, fs, _ = _sderivs(lambda ss: self._dp(ss, t), s, r0, r1)
            k = 1e-5
            ft = (self._dp(s, t + k) - self._dp(s, t - k)) / (2 * k)
            sr, stt = mp['s_r'], mp['s_t']
            shape = np.broadcast(sr, f0).shape
            f0, fs, ft = (np.broadcast_to(x, shape) for x in (f0, fs, ft))
            v = J(-f0, -fs * sr, -(ft + fs * stt))           # physical d/dr, d/dtheta at fixed r
            z = J(0 * f0, 0 * f0, 0 * f0)
            return dict(rr=v, tt=v, pp=v, rt=z)
        return tau


def parent_map(md, mv):
    mid = 0.5 * (mv.r0 + mv.r1)
    return [int(np.flatnonzero((md.r0 <= m) & (m <= md.r1))[0]) for m in mid]


def k_at(md, geo, st, sig, delta, K, N):
    ma = md.anelastic(sig * OMEGA_EARTH) if ANELASTIC else md
    mv = ma.viscous_core(sig * OMEGA_EARTH, delta=delta)
    par = parent_map(md, mv)
    g = _ByParent(geo, par)
    use_cs = CORE_STRESS and not isinstance(st, _RouteA)
    nh = (_CoreStress(st, par) if use_cs else _ByParent(st, par)) if st is not None else None
    import figure_choice as fc
    kw = {}
    if fc.FIGURE == 'level':        # N per kind of fluid sub-layer (sublayer_N_prem.py): 40 km layers 36, 5-delta 16
        from earthtides.solver import resolution_per_layer, viscous_core_resolution
        Kl, Nl = resolution_per_layer(mv, viscous_core_resolution(K, delta, N_bulk=N, N_wall=36, K=K, N=N))
        kw = dict(layer_K=Kl, layer_N=Nl)
    T = TidalSolver(mv, g, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', nonhydro=nh, **kw)
    T.solve()
    return complex(T.results()['kk']), T._solver_used, (T.__dict__.get('_block_info') or {}).get('residual')


def fcn(md, geo, st, delta, K, N):
    x = np.array(SIGMA)
    out = [k_at(md, geo, st, s, delta, K, N) for s in x]
    k = np.array([o[0] for o in out])
    c, b, d = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]).astype(complex), k * x, rcond=None)[0]
    misfit = np.abs((b * x - d) / (x - c) - k).max()
    return c, misfit, [o[1] for o in out], max((o[2] or 0) for o in out)


def main():
    cmb, K, N = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    deltas = [float(v) * 1e3 for v in sys.argv[4:]]
    run = Run(f'task C: FCN of the non-hydrostatic PREM, viscous core (CMB {cmb:.0f} m, route {ROUTE}, '
              f'core stress {CORE_STRESS}, core dp {CORE_DP}, anelastic {ANELASTIC})',
              params=dict(cmb=cmb, K=K, N=N, deltas=deltas, route=ROUTE, core_stress=CORE_STRESS, core_dp=CORE_DP,
                          anelastic=ANELASTIC,
                          sigma=SIGMA), inputs=['data/PREM.dat', 'data/PREMs.dat']).__enter__()
    lines = []

    def say(t):
        print(t, flush=True)
        lines.append(t)
    md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
    import figure_choice as fc
    base = fc.base(md)
    if cmb < 0:
        geo, st, tag = base, None, 'hydrostatic PREM'
    else:
        S = FlatSupport(md, mds, SHELLS, base, OMEGA_EARTH)
        a = S.fit({'J2': 1.14e-5, 'cmb': cmb})
        r = S.response(a)
        st = FlatState(S, a, base, OMEGA_EARTH)
        geo = FlatMap(base, st)
        tag = f'non-hydrostatic, CMB excess {cmb:.0f} m (e_f +{100 * r["e_f"] / 0.0025452:.2f} %), route {ROUTE}'
        if ROUTE == 'A':
            st = _RouteA(st)
    say(f'{tag}, viscous core, K {K}, N {N}, core dp {CORE_DP}' + (', anelastic' if ANELASTIC else ', elastic'))
    rows = []
    for dl in deltas:
        t = time.time()
        c, mis, used, res = fcn(md, geo, st, dl, K, N)
        rows.append((dl, c))
        say(f'  delta {dl / 1e3:5.2f} km: FCN {1 / (c.real - 1):7.2f} sd ({c.real:.7f}), Im {c.imag:+.2e} cpsd | '
            f'pole-fit misfit {mis:.0e} | {set(used)} residual {res:.0e} | {time.time() - t:5.1f} s')
    if len(rows) >= 2:
        d = np.array([r[0] for r in rows]) / 1e3
        p = np.array([1 / (r[1].real - 1) for r in rows])
        for deg in (1, 2):
            if len(rows) > deg:
                use = np.argsort(d)[:deg + 1]               # the smallest deltas
                coef = np.polyfit(d[use], p[use], deg)
                say(f'  delta -> 0, degree {deg} from {sorted(d[use])} km: FCN {np.polyval(coef, 0.0):7.2f} sd')
    name = 'hydro' if cmb < 0 else f'{cmb:.0f}m_{ROUTE}' + ('_cs' if CORE_STRESS and ROUTE == 'B' else '') + \
        ('_' + CORE_DP if CORE_STRESS and ROUTE == 'B' else '')
    name += '_anel' if ANELASTIC else ''
    import figure_choice as fc
    run.write_text(f'results/nonhydro_fcn_viscous_{name}_K{K}_N{N}{fc.SUF}.txt', '\n'.join(lines))
    run.__exit__(None, None, None)


if __name__ == '__main__':
    main()
