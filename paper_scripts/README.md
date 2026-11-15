# Reproducing the PEPI paper

Scripts and stored results for

> Bos, M. S., *Body tides of a flattened, rotating, non-hydrostatic Earth: a transparent numerical model*
> (Physics of the Earth and Planetary Interiors, submitted) and its Supplementary Material.

Every stored result carries its provenance in `results/<file>.meta.json` and one line in
`results/LEDGER.jsonl` (script and its sha256, commit, command line, environment, inputs, wall time, peak
memory, host).

## Running

```
python paper_scripts/run_all.py --list          # the jobs and what they produce
python paper_scripts/run_all.py                 # quick: numbers, tables and figures (about 20 minutes)
python paper_scripts/run_all.py --full          # + everything a laptop can do (a few hours)
python paper_scripts/run_all.py --heavy         # + the runs that need a large machine
python paper_scripts/run_all.py --installed     # with the installed package (pip install earth-tides)
```

Each job runs in a temporary copy of the tree, so `results/` is never overwritten, and its outputs are
compared with the stored ones number by number (`ok` or `DIFF`). `--keep DIR` keeps the new outputs,
`--update` copies them over the stored ones. The paper uses the level figure; the jobs that need it set
`EARTHTIDES_FIGURE=level` themselves. The jobs that compare with first-order theories (Smith 1974, Dehant
et al. 1999, the exact spheroid, the Hough and bump tests, the tests of the inviscid shell) use the
first-order figure, as the paper does. The `heavy` jobs need up to 24 GB and 16 cores for hours (they were
run on a 28 GB Linux machine); their stored results are the reference.

The stored results were computed on macOS with Apple's Accelerate library. On other platforms and BLAS
libraries the last printed digits can differ (up to a few 1e-6 in the Love numbers of the non-hydrostatic
runs, below the precision quoted in the paper), and the comparison reports them as `DIFF`. The release
workflow runs the quick jobs on Linux, macOS and Windows with `--report-only`: differences are listed, and only
a script that fails stops the release.

A few numbers rest on external programs or data: the IERS routine DEHANTTIDEINEL (`--external`,
`--iers DIR`, see `tools/iers/`) for "earth-tides reproduces the l(1) terms of DEHANTTIDEINEL to 1e-3 mm",
and the spherical transversely isotropic Love numbers of Bos & Scherneck (2013) for the anisotropy numbers of
Sect. S11 (`scripts/anisotropy_budget.py` reads them).

## Where each result comes from

| paper | jobs |
|---|---|
| numbers in the text not printed by another script (Ekman thicknesses, cot factor, Coriolis ratio, Voigt, ocean to rock, angular momentum at other spins, millimetres of the IERS latitude terms, Poisson jumps, resonance strength against Rosat et al. 2009) | `paper_numbers_v2` |
| Sect. 3.4, the level figure; Sect. 4 | `validate_level_figure`, `love_level_figure`, `am_wall_level`, `fcn_level_figure`, `angular_momentum`, `fcn_period` |
| Sect. 4, Table S15 (verification) | `check_hard`, `check_m`, `check_bump`, `exact_spheroid`, `rotating_spheroid`, `smith1974`, `ex3`-`ex6`, `quadrature_nodes`, `figS_level` (convergence) |
| S1-S2 (parametrization, conventions; Table S1) | `convention_table`, `frame_size` |
| S3 (comparison with Dehant et al. 1999 and the IERS Conventions; Tables S2-S4, Figs S1-S2) | `tables_viscous`, `table_anelastic`, `fig_ddw_iers`, `fig_dispersion`, `ddw_fingerprint`, `ddw_reprojection`, `ddw_wplus`, `l1_frame_sensitivity`, `wplus_spindown`, `compare_ddw1999`, `ddw_lowK`, `gravimetric_fcn430` |
| S4 (the sign in Smith's equations; Table S5) | `smith1974`, `smith1974_sign` |
| Sect. 5.1, Fig. 2 (the free core nutation, FICN) | `fig_fcn`, `fcn_level_figure` |
| Sect. 5.2, Figs 3-5 (beams, attractors, layered core) and their supplement figures | `figures_v2`, `fig_rays_buoyancy`, `fig_layered_core`, `fig_attractor3d`, `figS_core_flow_rays`, `figS_sphere_flat`, `rays_disc`, `rays_disc_layered`, `rays_hamiltonian`, `m2_rays_recheck`, `rays_m2_long`, `core_buoyancy`, `neutral_core`; maps: `attractor_*` (heavy) |
| Sect. 5.3, Figs 6-7, S5 (viscous core, slip term, the limit delta -> 0) | `fig_core`, `sweep_K8/16/24`, `core_sweep_offset`, `viscous_gap`, `slip_vs_viscous`, `slip_nonlevel_waves`, `hough_share_slipterm`, `exact_levels_share`, `delta_vs_limit`, `delta_fit_forms`, `ekman_profile`, `ekman_resolution`, `sublayer_N_prem`, `fcn_share_inviscid`; heavy: `viscous_limit_Kthin`, `viscous_limit_Kladder`, `fcn_extrapolation`, `sweep_K96`, `sweep_K168_*`, `fcn_noic_*` |
| S6 (the inviscid core: Table S8) | `b2_profile`, `b2_split`, `b2_wahr_h0`, `b2_ingredients_sphere`, `b2_ingredients_flat`, `b2_hough_coll`, `conv_deflated`, `conv_inviscid_lowK`, `conv_nocoriolis` |
| Sect. 6, Figs 8-10, S7 (non-hydrostatic Earth; Tables 1, S9-S13) | `nonhydro_figures`, `table_nonhydro_love`, `nonhydro_support`, `stress_split*`, `nonhydro_diurnal`, `nonhydro_am_sweep`, `nonhydro_am_viscous`, `nonhydro_fcn_viscous_*`, `core_rigidity`, `am_floor` |
| S11 (the Earth model) | `anisotropy_budget` |
| examples 7-9 | `ex7`, `ex8`, `ex9` |

`run_all.py --paper v2` and `--paper v1` hold the job lists of the earlier versions of the paper; their
scripts are not all part of this release.
