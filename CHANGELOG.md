# Changelog of earth-tides

## 1.1.1 (2026-11, with the manual version 1.1)

**Changed default: the stored Love-number tables.** `data/love_PREM.json` and `love_PREM_allwaves.json`
(`etide --love prem`, `prem-allwaves`) now hold the recommended values of the paper: anelastic PREM on the
level figure, made non-hydrostatic (excess J2 of Chambat et al. 2010, elastic support) so that the free core
nutation lies at the observed 431.4 sidereal days instead of the 459 of hydrostatic PREM. This changes h0 by
2.9e-3 at K1, 0.12 at psi1 and up to 0.25 at the small lines closest to the free core nutation;
outside 0.99-1.02 cycles per day by at most 2.5e-4. The set-up file is
`tools/stored_tables/love_PREM.toml`. The hydrostatic tables of 1.1.0 remain as `love_PREM_hydrostatic.json`
and `love_PREM_allwaves_hydrostatic.json` (`--love prem-hydrostatic`, `prem-allwaves-hydrostatic`).
`earthtides-love` itself still computes the hydrostatic Earth unless its set-up file has a
`[nonhydrostatic]` table. `etide` still defaults to the IERS Conventions.

The reproduction package of the paper is completed:

* The scripts of the heavy runs of the viscous core that the Supplementary Material quotes
  (`converge_resolution.py`, `share_vs_delta.py`, `strat_share_test.py`, `wall_fields.py`) are shipped with
  their stored results and provenance records; they are not rerun by `run_all.py` (core degree 48 to 96).
* Five new quick jobs: `table4_spread` (Table S13 against the core resolution and the linear solver),
  `ef_vs_cmb_fcn` (the free core nutation follows the excess dynamical ellipticity of the core, not the
  flattening of its boundary), `mathews1995_factor2` (the defining equation of Mathews et al. 1995a against
  the Conventions for m = 0, 1, 2, and their l(1) against ours), `table_recommended` (Table S1, the
  recommended Love numbers, an excerpt of the stored table) and `widmer_qm1_test` (the anelastic comparison
  with the shear Q of Widmer et al. 1991, as used by the Conventions).
* The anelastic comparison (Table S5) has its imaginary parts in a second table (Table S6).
* The paper scripts and examples use the exact frequencies of psi1 and phi1 (1.005475688 and 1.008213728
  cycles per day) instead of rounded ones; at psi1, 1e-7 cycles per day moves h0 by 1e-4.
* Windows: the paper scripts no longer need the POSIX module `resource`; a solver test allows for
  platform round-off.
* CI: the paper reproduction runs once, on macOS.

## 1.1.0 (2026-11, with the manual version 1.1)

The release that goes with the paper "Body tides of a flattened, rotating, non-hydrostatic Earth:
a transparent numerical model" (Bos, Physics of the Earth and Planetary Interiors, submitted).

**Changed default: the level figure.** The flattened Earth is now built from surfaces that are level
surfaces of its own reference potential (`figure='level'`, `geometry.figure(model, 'level')`), to all
orders in the flattening, instead of the first-order (Clairaut) map of 1.0. The free core nutation of
hydrostatic PREM moves from 457.3 to 456.2 sidereal days (elastic), angular momentum balances to 5e-9 of
the tidal torque instead of 8e-4, and away from the resonance the Love numbers change by at most 3e-5
(h0 at O1 in the anelastic table; 4.8e-3 at psi1, where the resonance moves).
The stored tables (`love_PREM.json`, `love_PREM_allwaves.json`) are recomputed with it. The first-order
figure remains an option (`--figure first-order`, `figure='first-order'`), for comparison with
first-order theories.

**New**

* A viscous liquid core with no-slip boundaries, Ekman sub-layers with their own Chebyshev nodes, and the
  limit of vanishing viscosity: the forced response of an inviscid core inside a shell with an inner core
  is ill-posed, the viscous limit is the reference.
* The slip condition of a fluid core at a boundary that is not a level surface (pressure term of the
  normal traction; `slip_nonlevel`, on by default).
* A non-hydrostatic Earth (`earthtides.nonhydro`; `[nonhydrostatic]` table of the set-up file): zonal
  mantle loads supported elastically or by instantaneous viscous flow, fitted to targets (excess J2,
  flattening of the core-mantle boundary or the surface, excess dynamical ellipticity of the core), with
  the deviatoric stress of the support and the slip terms of the non-level boundaries in the tidal
  equations.
* Robust Love-number tables: `earthtides-love` solves every wave with N = 16, 20, 24 and 30 in the liquid
  core and takes the median of each parameter (`--core-median`, `core_median`; `--single-core` for one
  solve). An inviscid core puts pseudo-modes on some waves at any resolution, on different waves at each;
  the median rejects them, and `core_spread` in every entry records the deviation. The stored tables are
  made this way.
* `love.Options(t0=...)`, `earthtides-love --t0`: the reference period of the Earth model's moduli (1 s for PREM; 200 s
  reproduces the dispersion assumed for the anelastic values of the IERS Conventions).
* The scalar equations of Smith (1974), first order in the flattening (`earthtides.smith1974`), with the
  corrected sign of the radial rho_2 term of his eq. (5.26).
* Solvers for large angular resolution: dense layer blocks, banded coupling between degrees
  (`band=`), a direct layer-by-layer solver (`linear_solver='layers'`) and a pressure unknown in the
  fluid for thin Ekman layers.
* `paper_scripts/run_all.py`: the jobs of the PEPI paper (`--paper v3`, default), per-job environments,
  `--report-only` for platforms other than the reference machine.
* Examples ex7 (the non-hydrostatic Earth), ex8 (towards the viscous limit) and ex9 (level against
  first-order figure).

**Fixed**

* `etide` now applies h', l' and the zonal l(1) of a Love table (1.0 used them only from the IERS
  model), so a table of `earthtides-love` and the IERS model are treated alike.
* A liquid layer is a true fluid only when its shear modulus is exactly zero (1.0: |mu| < 1 Pa, which
  turned a very weakly viscous core into an inviscid one).
* Static runs (weak-solid core, rigid-rotation constraints) solve the border system by an equilibrated
  Schur complement and route rotation regions to SuperLU; the result is the same, the conditioning better.

**Faster, with less memory**

* Sparse assembly, dense layer blocks and a block-tridiagonal LAPACK solver with iterative refinement
  (a K8/N44 solve about 4 s on a laptop, was 15 s); per-layer K and N (`--resolution FILE`,
  `--fluidcore_N`, `--fluidcore_K`).

## 1.0.0 (2026-10-10)

First release: Love numbers of a rotating, flattened (first-order figure), layered, anelastic,
self-gravitating Earth with an inviscid liquid core; `etide`, `earthtides-love`, `earthtides-examples`;
examples ex1-ex5 and the manual version 1.0.
