# Reproducing the PEPI paper

Scripts and stored results for

> Bos, M. S., *Body tides of a flattened, rotating, non-hydrostatic Earth: a transparent numerical model*
> (in preparation for Physics of the Earth and Planetary Interiors) and its Supplementary Material.

Every stored result carries its provenance in `results/<file>.meta.json` and one line in
`results/LEDGER.jsonl` (script and its sha256, commit, command line, environment, inputs, wall time, peak
memory, host).

## Running

```
python paper_scripts/run_all.py --list          # the jobs and what they produce
python paper_scripts/run_all.py                 # quick: numbers, tables and figures (about 1.5 hours; example 8 alone 40 minutes)
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

Four scripts of heavy runs that the Supplementary Material quotes are shipped with their stored results but
are not jobs of `run_all.py`: `scripts/converge_resolution.py` (convergence of the viscous core in band, N
and K; `results/converge_resolution_*.txt`), `scripts/share_vs_delta.py` (resonant shares per Ekman-layer
thickness; `results/share_vs_delta_*`), `scripts/strat_share_test.py` (stratified, neutral and uniform
core; `results/strat_share_test.*`) and `scripts/wall_fields.py` (fields next to the core-mantle boundary,
Hough's Earth; `results/wall_fields_*`, the `.npz` maps only in the Zenodo archive). Each result has a
`.meta.json` record with the command and arguments of its run; rerun a result with that command.

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

Numbering of the current draft (main_v4, supplement_v4). `docs/paper_v4_numbers_inventory.md` maps every
number in the text to its script and stored result; `run_all.py --list` gives the same location per job.

| paper | jobs |
|---|---|
| numbers in the text not printed by another script (Ekman thicknesses, cot factor, Coriolis ratio, Voigt, ocean to rock, angular momentum at other spins, millimetres of the IERS latitude terms, Poisson jumps) | `paper_numbers_v2` |
| Sect. 3, Fig. 1 (method, the level figure, the conventions) | `fig_grid`, `quadrature_nodes`, `validate_level_figure`, `love_level_figure`, `frame_size`, `convention_table`, `ex3` |
| Sect. 4 and S10, Table S16, Fig. S8 (accuracy, verification, convergence) | `check_hard`, `check_m`, `check_bump`, `exact_spheroid`, `rotating_spheroid`, `smith1974`, `ex3`-`ex6`, `angular_momentum`, `am_wall_level`, `table4_spread`, `figS_level` |
| Sect. 5.1, Table 1, Fig. 2; S5.1 (the free core nutation, FICN) | `fig_fcn`, `fcn_level_figure`, `gravimetric_fcn430` |
| Sect. 5.2, Fig. 3; S5.2, Figs S3-S5 (beams, attractors, stratification) | `figures_v2`, `fig_attractor3d`, `fig_rays_buoyancy`, `figS_core_flow_rays`, `rays_disc`, `rays_disc_layered`, `rays_hamiltonian`, `core_buoyancy`, `neutral_core`, `attractor_control`, `m2_rays_recheck`, `rays_m2_long`; maps: `attractor_*` (heavy) |
| Sect. 5.3, Fig. 4; S5.3-S5.5, Fig. S6, Table S7 (viscous core, Ekman layer, the limit delta -> 0, slip term) | `fig_core`, `sweep_K8/16/24`, `core_sweep_offset`, `viscous_gap`, `slip_vs_viscous`, `slip_nonlevel_waves`, `hough_share_slipterm`, `exact_levels_share`, `delta_vs_limit`, `delta_fit_forms`, `viscous_limit_uncertainty`, `table_slipshare` (Table S7; `fcn_share_level*`), `wall_fields_un_check`, `wall_pressure_check`, `ekman_profile`, `ekman_resolution`, `fcn_share_inviscid`, `ex6`, `ex8`; heavy: `viscous_limit_Kthin`, `viscous_limit_Kladder`, `fcn_extrapolation`, `sweep_K96`, `sweep_K168_*`, `fcn_noic_inviscid` |
| S5.6, Fig. S7, Table S8 (latitude dependence) | `figS_level`, `table_latitude`, `tables_viscous` |
| S6, Table S9 (the inviscid core) | `b2_profile`, `b2_split`, `b2_wahr_h0`, `b2_ingredients_sphere`, `b2_ingredients_flat`, `b2_hough_coll`, `conv_deflated`, `conv_inviscid_lowK`, `conv_nocoriolis`, `ddw_lowK` |
| Sect. 6, Figs 5-7; S7, Tables S10-S14 (non-hydrostatic Earth) | `nonhydro_figures`, `table_nonhydro_love`, `table4_spread`, `ef_vs_cmb_fcn`, `nonhydro_support`, `stress_split*`, `nonhydro_diurnal`, `nonhydro_am_sweep`, `nonhydro_am_viscous`, `nonhydro_fcn_viscous_*`, `core_rigidity`, `am_floor`, `ex7`, `ex9` |
| Sect. 7, Table 2 (comparison, error budget in mm) | `displacement_scale`, `stress_same_period`, `figure_same_period` (with `table_nonhydro_love_fo`), `viscous_limit_uncertainty`, `ddw_fingerprint`, `l1_frame_sensitivity`, `slip_nonlevel_waves`, `tables_viscous`, `table_anelastic` |
| S1, Table S1 (recommended Love numbers); S2, Table S2 (conventions) | `table_recommended`, `ddw_reprojection`, `convention_table`, `frame_size` |
| S3, Tables S3-S5, Figs S1-S2 (Dehant et al. 1999 and the IERS Conventions) | `table_ddw` (Table S3 from `tables_viscous_fo` and `fcn_extrapolation_fo`), `tables_viscous`, `table_anelastic`, `fig_ddw_iers`, `fig_dispersion`, `ddw_fingerprint`, `ddw_reprojection`, `ddw_wplus`, `wplus_spindown`, `mathews1995_factor2`, `widmer_qm1_test`, `displacement_scale`, `fcn_extrapolation` |
| S4, Table S6 (the sign in Smith's equations) | `smith1974`, `smith1974_sign` |
| S9 (comparison of methods) | `ex4`, `ex8`, `viscous_limit_Kthin` |
| S11, Table S17 (the Earth model) | `anisotropy_budget`, `check_prem_table1` |
| examples 7-9 | `ex7`, `ex8`, `ex9` |

Jobs whose result the v4 text no longer quotes (background of earlier versions, still reproduced):
`fcn_period`, `fig_layered_core`, `figS_sphere_flat`, `sphere_flat_ratios`, `depth_sensitivity`,
`compare_ddw1999`, `fcn_noic_viscous`, `attractor_layered`, `attractor_sphere`.

`run_all.py --paper v2` and `--paper v1` hold the job lists of the earlier versions of the paper; their
scripts are not all part of this release.
