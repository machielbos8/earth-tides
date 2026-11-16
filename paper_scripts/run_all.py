#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Reproduce the numbers, tables and figures of the earth-tides paper.
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
# 10/10/2026 Machiel Bos
#===============================================================================
"""Reproduce the results of

    Bos, M. S., "Solid Earth tides of a flattened, rotating, anelastic Earth:
    a transparent numerical model", Journal of Geodesy (submitted), and its Online Resource 1.

Every job runs one script of scripts/ (or the figure script, or a worked example) in a temporary
copy of this tree, so the stored results in results/ are never overwritten: they are the reference.
The outputs of a job are then compared with the stored ones number by number: a printed number may
differ by one unit in its last digit (or 1e-8 relative); the text must be equal. Jobs that solve an
ill-posed problem on purpose (ILL_POSED) are run but not compared ('ill'). README.md lists
which job produces which table, figure and number of the paper.

Levels
  --quick     (default) the fast jobs and all figures redrawn from the stored data, ~3 minutes
  --full      every job that needs only this package (Python, NumPy, SciPy, Matplotlib), ~1.5 h
  --external  in addition the jobs that need data or programs not distributed here; each is
              skipped, with the reason, if its inputs are not found:
                ocean tide grids (FES2014b, FES2022b, GOT5.5, TPXO10-atlas) in --grids DIR
                the IERS routine DEHANTTIDEINEL, compiled in variants, and the DE421 ephemeris
                  in --iers DIR (see tools/iers/), with skyfield installed
                the FES2004 coefficients of the IERS Conventions (fes2004.dat) in --iers DIR
              The ocean tide loading at the gravimeter stations (results/tidal_gravity/CARGA_*.csv)
              was computed with CARGA, which is not distributed; those files are inputs here.
Other options
  --list      list the jobs and exit
  --only NAME run only the named jobs (and nothing else)
  --keep DIR  keep the outputs in DIR (default: a temporary directory that is removed)
  --installed use the installed package (pip install earth-tides) instead of src/
  --update    copy the new outputs over the stored results (to refresh them after a code change)
  --report-only  differences are reported, not fatal (CI on platforms other than the reference machine)
  --paper v2  the jobs of the PEPI paper (main_v2, supplement_v2); --heavy adds the runs that need a
              large machine (28 GB), whose stored results carry their run id in results/*.meta.json
Usage: python paper_scripts/run_all.py [--paper v2] [--quick|--full|--external|--heavy] [--only NAME ...]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# name, level, command (relative to the tree), outputs, requirement, paper items
#   (OR = Online Resource 1, named by section title so that renumbering does not break the labels)
#   level: quick | full | external;  requirement: None | 'grids' | 'iers' | 'fes2004'
#   an output 'stdout:NAME' is the printed output, stored as results/stdout/NAME.txt
JOBS = [
    # --- method and verification (Sects 3-4, Table 1, Online Resource: verification)
    ('check_hard', 'quick', ['scripts/check_hard.py'], ['stdout:check_hard'], None,
     'Table 1: PREM as a sphere vs Bos & Scherneck (2013)'),
    ('check_m', 'quick', ['scripts/check_m.py'], ['stdout:check_m'], None,
     'Table 1: independence of m'),
    ('check_bump', 'quick', ['scripts/check_bump.py'], ['stdout:check_bump'], None,
     'Table 1, OR verification: distorted mesh'),
    ('exact_spheroid', 'quick', ['scripts/check_exact_spheroid.py'], ['results/exact_spheroid.txt'],
     'sympy', 'Table 1, OR verification: homogeneous elastic spheroid, cost f^2/2'),
    ('smith1974', 'quick', ['scripts/check_smith1974.py'], ['results/smith1974.txt'], 'sympy',
     'OR verification: Smith (1974) equations vs exact series and solver, m = 1 and 2 (w+ included)'),
    ('rotating_spheroid', 'quick', ['scripts/check_rotating_spheroid.py'], ['results/rotating_spheroid.txt'],
     None, 'OR verification: rotating homogeneous spheroid, flattening and Coriolis parts of w+'),
    ('ex3', 'quick', ['src/earthtides/examples/ex3/homogeneous.py'],
     ['src/earthtides/examples/ex3/homogeneous.out'], None, 'Table 1: homogeneous spheres'),
    ('ex4', 'full', ['src/earthtides/examples/ex4/greff2005.py'],
     ['src/earthtides/examples/ex4/greff2005.out'], None, 'Table 1, OR verification: Greff-Lefftz et al. (2005)'),
    ('ex5', 'quick', ['src/earthtides/examples/ex5/hough1895.py'],
     ['src/earthtides/examples/ex5/hough1895.out'], None, "Table 1, Sect. 6.1: Hough's fluid core"),
    ('angular_momentum', 'quick', ['scripts/check_angular_momentum.py'], ['stdout:check_angular_momentum'],
     None, 'Table 1, OR verification: angular-momentum balance'),
    ('fcn_period', 'quick', ['scripts/fcn_period.py', 'anelastic'], ['stdout:fcn_period_anelastic'],
     None, 'Sect. 6.1, OR FCN period: FCN of anelastic hydrostatic PREM (460.0 sidereal days)'),
    ('convention_table', 'quick', ['scripts/convention_table.py'], ['results/convention_table.txt'],
     None, 'OR latitude terms/conventions: effect of the conventions'),
    ('frame_size', 'quick', ['scripts/frame_size.py'], ['results/frame_size.txt'],
     None, 'Sect. 3.5, OR conventions: size of the removed rotation and of the frame choice (mm)'),
    ('admittance_convention', 'quick', ['scripts/check_admittance_convention.py'],
     ['stdout:check_admittance_convention'], None,
     'OR ocean loading: synthetic equilibrium ocean gives A21 = +1; no convention maps the ocean lag onto the IERS'),
    ('check_iers_tables', 'quick', ['scripts/check_iers_tables.py'], ['stdout:check_iers_tables'], None,
     'OR how the Conventions compute + ocean loading: IERS Tables 7.2, 7.3a reproduced only with the empirical admittance'),
    ('pm_resonance', 'quick', ['scripts/check_pm_resonance.py'], ['stdout:check_pm_resonance'], None,
     'OR ocean loading (nutation test): polar-motion resonance of the nutations with our and the IERS admittance sign'),
    ('anisotropy_budget', 'quick', ['scripts/anisotropy_budget.py'], ['stdout:anisotropy_budget'], None,
     'OR transverse isotropy: transverse isotropy vs flattening, M2 displacement (mm)'),
    ('ddw_fingerprint', 'quick', ['scripts/ddw_fingerprint.py'], ['results/ddw_fingerprint.txt'], None,
     'Sect. 5.1, OR comparison with Dehant et al.: fingerprints of reference-state changes vs the offset from Dehant et al. (1999), mm'),
    ('ddw_reprojection', 'quick', ['scripts/ddw_reprojection.py'], ['results/ddw_reprojection.txt'], None,
     'OR comparison with Dehant et al.: their Table 10 re-projected onto Table 11; l(1) is in their solution (w+), not in the conversion'),
    ('ddw_wplus', 'quick', ['scripts/ddw_wplus.py'], ['results/ddw_wplus.txt'], None,
     'OR comparison with Dehant et al.: w+ by line, split as in their Table 4, and by order (2 min)'),
    ('figures', 'quick', ['paper/make_figures.py', '--replot'], ['figures'], None,
     'Figs 1-5 and the OR figures (from the stored data)'),
    # --- Love numbers and the free core nutation (Sects 5-6, Tables 2-4)
    ('diurnal', 'full', ['scripts/run_diurnal.py'], ['results/diurnal_PREM.txt', 'results/figdata_fcn.npz'],
     None, 'Table 2, Fig. 4: diurnal band, resonance sweep'),
    ('compare_models', 'full', ['scripts/compare_models.py'], ['results/compare_models.txt'], None,
     'Sect. 5.3: PREM vs STW105'),
    ('anelastic', 'full', ['scripts/run_anelastic.py'], ['results/anelastic_PREMs.txt'], None,
     'Sect. 5.3: anelastic change of the latitude terms'),
    ('coriolis', 'full', ['scripts/run_coriolis.py'], ['results/coriolis_PREMs.txt'], None,
     'Sect. 5.2: Coriolis part of l(1) and l\''),
    ('scan_delta_plus', 'full', ['scripts/scan_delta_plus.py'], ['stdout:scan_delta_plus'], None,
     'not in the paper (former tidal-gravity test): gravimetric factors vs Dehant et al. (1999)'),
    ('gravimetric_spinup', 'full', ['scripts/gravimetric_factor_spinup.py'],
     ['results/gravimetric_factor_spinup.txt'], None, 'Sect. 6.1, OR FCN period + resonance formula: FCN at 431.3 days by spin-up; K1 load resonance unchanged'),
    ('gravimetric_fcn430', 'full', ['scripts/gravimetric_factor_fcn430.py'],
     ['results/gravimetric_factor_fcn430.txt'], None, 'OR FCN period: FCN residue of the gravimetric factor (4.41e-5) vs Rosat et al. (2009)'),
    ('k1o1_sensitivity', 'full', ['scripts/k1o1_sensitivity.py'], ['results/k1o1_sensitivity.txt'],
     None, 'not in the paper (former tidal-gravity test): K1/O1 vs mantle rigidity'),
    ('love_table', 'full', ['-m', 'earthtides.love', '--jobs', '4', '-o', 'results/love_PREM_check.json'],
     ['results/love_PREM_check.json'], None, 'Sect. 7: default Love table of PREM (data/love_PREM.json)'),
    ('displacement_budget', 'full', ['scripts/displacement_budget.py'], ['results/displacement_budget.txt'],
     None, 'Sect. 7.1, OR line-by-line difference: effect of each choice at four stations'),
    ('prem_iers_harmonics', 'full', ['scripts/prem_iers_harmonics.py'], ['results/prem_iers_harmonics.txt'],
     None, 'Sect. 7.1: PREM - IERS, rms and line amplitudes'),
    ('paper_numbers', 'full', ['scripts/paper_numbers.py'], ['results/paper_numbers.txt'], None,
     'numbers in the text not printed elsewhere (FCN periods, Table 4 residues, mesh, mm conversions)'),
    ('figure_data', 'full', ['paper/make_figures.py'], ['results/figdata.npz', 'results/figdata_toy.npz'],
     None, 'Figs 1-5 and the OR figures with their data recomputed'),
    # --- no longer in the paper: the tidal-gravity test of the admittance sign (replaced by the nutation
    #     test of Sect. 6.4) and the admittance from Chao et al. (1996); kept for the revision and the
    #     discussion with the IERS Conventions Centre
    ('check_chao1996', 'quick', ['scripts/check_chao1996.py'], ['stdout:check_chao1996'], None,
     'not in the paper: admittance from the angular momenta of Chao et al. (1996)'),
    ('ducarme', 'quick', ['scripts/check_tidal_gravity_ducarme.py'],
     ['results/tidal_gravity/check_ducarme.txt'], None, 'not in the paper: K1/O1 ratios of Ducarme et al.'),
    ('tidal_gravity_k1', 'quick', ['scripts/check_tidal_gravity_k1.py'],
     ['stdout:check_tidal_gravity_k1'], None, 'not in the paper: K1/O1 with our ocean loading (CARGA inputs)'),
    # --- jobs that need external data or programs
    ('otl_fcn_resonance', 'external', ['scripts/otl_fcn_resonance.py'], ['results/otl_fcn_resonance.txt'],
     'grids', 'Sect. 6.2, Table 4, OR resonance formula + ocean loading: load resonance, admittance A21, resonant correction (FES2022b)'),
    ('station_error_budget', 'external', ['scripts/station_error_budget.py'],
     ['results/station_error_budget.txt', 'results/station_error_budget.npz'], 'grids',
     'Sect. 7.2, Fig. 5, Table 5: anelasticity and complete model - IERS at nine stations (ocean models and sign in OR ocean loading)'),
    ('iers_admittance', 'external', ['scripts/check_iers_admittance.py', '{IERS}/fes2004.dat'],
     ['stdout:check_iers_admittance'], 'fes2004', 'OR ocean loading: FES2004 admittance in the IERS conventions'),
    ('station_admittance', 'external', ['scripts/station_admittance.py'], ['results/station_admittance.npz'],
     'iers', 'input of body_tide_check: admittance per HARDISP line (25 minutes)'),
    ('body_tide_check', 'external', ['scripts/body_tide_check.py', '{IERS}', 'all'],
     ['results/body_tide_check.txt'], 'iers', 'OR verification against the IERS model: IERS degree 2 and 3, step 2'),
    ('etide_vs_iers', 'external', ['scripts/etide_vs_iers.py', '{IERS}'], ['results/etide_vs_iers.txt'],
     'iers', 'Sect. 7, OR verification against the IERS model: etide vs DEHANTTIDEINEL term by term'),
]

# ---------------------------------------------------------------------------------------- P1 v2 (PEPI)
# Bos, M. S., Tidal Love numbers of a flattened, rotating Earth with a viscous liquid core (PEPI; main_v2.tex
# and supplement_v2.tex). docs/paper_v2_numbers_inventory.md maps every number to its job. Jobs reused from
# the J Geod paper are taken from JOBS by name; 'heavy' jobs need a large machine (28 GB, 16 cores)
# and run only with --heavy; their stored results carry the run id in results/*.meta.json.
_V1 = {j[0]: j for j in JOBS}
_FIG = lambda *names: [f'figure:paper_pepi/figures/{n}.pdf' for n in names]
JOBS_V2 = [_V1[n] for n in ('check_hard', 'check_m', 'check_bump', 'exact_spheroid', 'smith1974',
                            'rotating_spheroid', 'ex3', 'ex5', 'angular_momentum', 'fcn_period',
                            'convention_table', 'frame_size', 'anisotropy_budget', 'ddw_fingerprint',
                            'ddw_reprojection', 'ddw_wplus')] + [
    # --- numbers in the text, tables, figures from stored data (quick)
    ('paper_numbers_v2', 'quick', ['scripts/paper_numbers_v2.py'], ['results/paper_numbers_v2.txt'], None,
     'numbers in the text not printed by another script (analytic, S1 gravity, 3.4, S5, S9, S3, S1, S12, 5.1)'),
    ('quadrature_nodes', 'quick', ['scripts/quadrature_nodes_test.py'], ['results/quadrature_nodes_test.txt'],
     None, 'Sect. 3.2: Gauss-Legendre nodes of the projection'),
    ('table_latitude', 'quick', ['paper_pepi/make_table_latitude.py'], ['results/paper1_table_latitude.txt'],
     None, 'Table S8 (flattening only, Coriolis only rows; inviscid core)'),
    ('tables_viscous', 'quick', ['paper_pepi/make_tables_viscous.py'], ['results/paper1_tables_viscous.txt'],
     None, 'Table S3 (Mf, O1, M2 rows), Table S8 (both row): viscous core, delta -> 0'),
    ('table_nonhydro_love', 'quick', ['paper_pepi/make_table4.py'], ['results/paper1_table4.txt'], None,
     'Tables S12 and S7 (non-hydrostatic Love numbers, elastic and viscous support)'),
    ('nonhydro_figures', 'quick', ['paper_pepi/make_figures.py'],
     ['results/fig_nonhydro_data.txt', 'results/figdata_nonhydro.npz']
     + _FIG('fig_nonhydro_state', 'figS_nonhydro_state_viscous', 'fig_fcn_ef', 'fig_am_sweep'), None,
     'Figs 8-10, S4; Sect. 6.2 (300 m, 27 MPa), Table S10 (FCN periods)'),
    ('core_sweep_offset', 'quick', ['scripts/core_sweep_offset.py'], ['results/core_sweep_offset.txt'], None,
     'Sect. 5.3: offsets of the viscous sweep'),
    ('viscous_gap', 'quick', ['scripts/fig4_viscous_gap.py'], ['results/fig4_viscous_gap.txt'], None,
     'Sect. 5.3, S11: viscous minus inviscid h(2)'),
    ('delta_vs_limit', 'quick', ['scripts/viscous_delta_vs_limit.py', '12'],
     ['results/viscous_delta_vs_limit_K12.txt'], None, 'S11: finite delta against the limit delta -> 0'),
    ('rays_disc', 'quick', ['scripts/rays_disc_grid.py'], ['results/rays_disc_grid.txt', 'results/rays_disc_grid.npz'],
     None, 'Figs 4-5: regions where no wave travels (PREM)'),
    ('rays_disc_layered', 'quick', ['scripts/rays_disc_grid.py', '--core', 'layered'],
     ['results/rays_disc_grid_layered.npz'], None, 'Fig. 5: the same for the layered core'),
    ('figures_v2', 'quick', ['paper_pepi/make_fig_characteristics.py'], _FIG('fig_characteristics'), None,
     'Fig. 3: characteristics'),
    ('fig_rays_buoyancy', 'quick', ['paper_pepi/make_fig_rays_buoyancy.py'], _FIG('fig_rays_buoyancy'), None,
     'Fig. 4 (stored core_flow_attractor maps)'),
    ('fig_layered_core', 'quick', ['paper_pepi/make_fig_layered_core.py'], _FIG('fig_layered_core'), None,
     'Fig. 5 (stored maps)'),
    ('fig_core', 'quick', ['paper_pepi/make_fig_core.py'], _FIG('fig_core_sweep', 'fig_core_ekman'), None,
     'Figs 6-7 (stored sweeps)'),
    ('fig_grid', 'quick', ['paper_pepi/make_fig_grid.py'], _FIG('fig_grid'), None,
     'Fig. 1: level-surface coordinates, the quarter the solver computes (from results/figdata.npz)'),
    ('fig_attractor3d', 'quick', ['paper_pepi/make_fig_attractor3d.py'], _FIG('figS_attractor3d'), None,
     'Fig. S5'),
    ('figS_core_flow_rays', 'quick', ['paper_pepi/make_figS_core_flow_rays.py'], _FIG('figS_core_flow_rays'),
     None, 'Fig. S6 (stored maps)'),
    ('figS_sphere_flat', 'quick', ['paper_pepi/make_figS_sphere_flat.py'], _FIG('figS_sphere_flat'), None,
     'Fig. S7 (stored maps)'),
    _V1['figures'],
    # --- full: everything that needs only this package, minutes to hours on a laptop
    ('smith1974_sign', 'full', ['scripts/smith1974_sign_test.py'], ['results/smith1974_sign_test.txt'], None,
     'S5.1: sign of the radial rho2 term of Smith (1974) eq. 5.26 (25 min)'),
    _V1['ex4'], _V1['gravimetric_fcn430'],
    ('ex6', 'full', ['src/earthtides/examples/ex6/ekman.py'], ['src/earthtides/examples/ex6/ekman.out'], None,
     'Sect. 5.3, Table S2: viscous Hough core against Greenspan'),
    ('b2_profile', 'full', ['scripts/b2_profile.py'], ['stdout:b2_profile'], None,
     'S13: radial half-wavelengths of the truncated core equations'),
    ('b2_split', 'full', ['scripts/b2_split.py'], ['stdout:b2_split'], None, 'S13: radial h-refinement'),
    ('b2_wahr_h0', 'full', ['scripts/b2_wahr_h0.py'], ['stdout:b2_wahr_h0'], None,
     "S13: Wahr's truncation, solver spread in h(0)"),
    ('b2_ingredients_sphere', 'full', ['scripts/b2_ingredients.py'], ['stdout:b2_ingredients_sphere'], None,
     'Table S12 (spherical rows)'),
    ('b2_ingredients_flat', 'full', ['scripts/b2_ingredients.py', 'flat'], ['stdout:b2_ingredients_flat'], None,
     'Table S12 (flattened rows)'),
    ('b2_hough_coll', 'full', ['scripts/b2_hough_coll.py'], ['stdout:b2_hough_coll'], None,
     "Table S12 (Hough's Earth)"),
    ('nonhydro_am_sweep', 'full', ['scripts/nonhydro_am_sweep.py', '450', '4'],
     ['results/nonhydro_am_sweep_450m.txt', 'results/nonhydro_am_sweep_450m.npz'], None,
     'Fig. 10, Table S10: angular momentum through the diurnal band (450 m)'),
    ('rays_hamiltonian', 'full', ['scripts/rays_hamiltonian.py', '1e-3', '4'], ['results/rays_hamiltonian_h0.001.txt'],
     None, 'Figs 4-5: attractor loops traced with Hamilton\'s equations (30 min)'),
    ('nonhydro_am_viscous', 'full', ['scripts/nonhydro_am_viscous.py', '10', '30', '1', '0.5'],
     ['results/nonhydro_am_viscous_K10_N30.txt'], None, 'S12: angular momentum with a viscous core (25 min, 4 GB)'),
    # --- heavy: a large machine (28 GB), hours
    ('fcn_extrapolation', 'heavy', ['scripts/fcn_resonance_extrapolation_v3.py', '12', '8', '--fresh'],
     ['results/fcn_resonance_extrapolation_v3_K12.txt'], None,
     'Table S3 (P1 to OO1), S11: FCN resonance at delta -> 0, core K 12 (26 min on 8 workers, 16 GB)'),
    ('nonhydro_fcn_viscous', 'heavy', ['scripts/nonhydro_fcn_viscous.py', '450', '10', '30', '2', '1', '0.5'],
     ['results/nonhydro_fcn_viscous_450m_B_cs_static_K10_N30.txt'], None,
     'Table S10, Fig. 9: FCN with a viscous core, 450 m (3.3 h; 0, 225 and 500 m likewise)'),
    ('attractor_K168', 'heavy', ['scripts/core_flow_attractor.py', '0.9175:PREM:0.5:168:16:15',
                                 '0.9175:PREM:0.25:168:16:15', '0.9075:neutral:0.5:168:16:15',
                                 '0.9075:neutral:0.25:168:16:15', '--band', '8', '--solid-P', '0', '--max-gb', '24',
                                 '--full-maps', '--tag', 'K168_S16N15'],
     ['results/core_flow_attractor_K168_S16N15.txt'], None, 'Fig. 4, S6, S7: beam maps at 0.9175 (37 min)'),
    ('attractor_M2', 'heavy', ['scripts/core_flow_attractor.py', '1.92700:PREM:0.25:168:16:15',
                               '1.92700:PREM:1:96:8', '--band', '8', '--solid-P', '0', '--max-gb', '24',
                               '--full-maps', '--tag', 'M2'],
     ['results/core_flow_attractor_M2.txt'], None, 'Fig. 4b, S7: beam maps at M2 (17 min)'),
    ('attractor_layered', 'heavy', ['scripts/core_flow_attractor.py', '0.9175:layered:0.25:168:16:15',
                                    '1.92700:layered:0.25:168:16:15', '--tag', 'layered', '--band', '8',
                                    '--solid-P', '0', '--full-maps', '--max-gb', '24'],
     ['results/core_flow_attractor_layered.txt'], None, 'Fig. 5: layered core (17 min)'),
    ('attractor_sphere', 'heavy', ['scripts/core_flow_attractor.py', '0.9175:PREM:0.25:168:16:15',
                                   '1.92700:PREM:0.25:168:16:15', '--tag', 'sphere', '--sphere', '--band', '8',
                                   '--solid-P', '0', '--full-maps', '--max-gb', '24'],
     ['results/core_flow_attractor_sphere.txt'], None, 'Fig. S7, Sect. 5.3: a sphere (23 min)'),
    ('attractor_scan', 'heavy', ['scripts/core_flow_attractor.py'] + [f'{f}:{c}:{d}:96:8' for f, c in
                               (('0.9175', 'PREM'), ('0.9075', 'neutral'), ('0.9125', 'neutral'))
                               for d in ('1', '0.5', '0.25')] + ['--tag', 'scan'],
     ['results/core_flow_attractor_scan.txt'], None, 'Fig. S6, S11: attractor energy against delta (47 min)'),
    ('sweep_K96', 'heavy', ['scripts/convergence_viscous_coreK.py', '96', '30', '1', '2', '0.0005', '--band', '12'],
     ['results/convergence_viscous_coreK96_N30_d1_s0.0005_b12.txt'], None, 'Fig. 6: diurnal sweep, core K 96 (3.4 h)'),
    ('sweep_K168_d0.5', 'heavy', ['scripts/convergence_viscous_coreK.py', '168', '30', '0.5', '1', '--band', '8',
                                  '--split', '16', '--interior-N', '15', '--solid-P', '0'],
     ['results/convergence_viscous_coreK168_N30_d0.5_b8_S16N15_sP0.txt'], None, 'Sect. 5.3: hump at delta 0.5 km (5.3 h)'),
    ('sweep_K168_d0.25', 'heavy', ['scripts/convergence_viscous_coreK.py', '168', '30', '0.25', '1', '--band', '8',
                                   '--split', '16', '--interior-N', '15', '--solid-P', '0'],
     ['results/convergence_viscous_coreK168_N30_d0.25_b8_S16N15_sP0.txt'], None, 'Sect. 5.3: hump at delta 0.25 km (5.3 h)'),
    ('sweep_K96_d1_split', 'heavy', ['scripts/convergence_viscous_coreK.py', '96', '30', '1', '1', '--band', '8',
                                     '--split', '16', '--interior-N', '15', '--solid-P', '0'],
     ['results/convergence_viscous_coreK96_N30_d1_b8_S16N15_sP0.txt'], None, 'Sect. 5.3: delta 1 km, split core (1.6 h)'),
    # --- added 2026-11-10 (review): scripts behind numbers that had no job
    ('stress_split', 'quick', ['scripts/nonhydro_stress_split.py', '450', '4'],
     ['results/nonhydro_stress_split_450m.txt'], None, 'Sect. 6.3, Table S10: tau and slip terms separately'),
    ('am_floor', 'quick', ['scripts/am_floor_spheroid.py'], ['results/am_floor_spheroid.txt'], None,
     'Sect. 6.3, S10: angular-momentum floor = first-order figure (homogeneous spheroid)'),
    ('sphere_flat_ratios', 'quick', ['scripts/sphere_flat_ratios.py'], ['results/sphere_flat_ratios.txt'], None,
     'S9: sphere / flattened beam energy'),
    ('core_rigidity', 'quick', ['scripts/nonhydro_core_rigidity.py'], ['results/nonhydro_core_rigidity.txt'], None,
     'Sect. 6.2: weak-solid core rigidity of the static support (< 1e-5)'),
    ('ekman_profile', 'quick', ['scripts/ekman_profile.py'], ['results/ekman_profile.npz', 'results/ekman_profile.txt'],
     None, 'Fig. 7a, S9: the Ekman layer against the local solution'),
    ('attractor_control', 'quick', ['scripts/core_flow_attractor_control.py', 'scan'],
     ['results/core_flow_attractor_scan_control.txt'], None, 'S9: turned-loop control 0.95-1.05'),
    ('core_buoyancy', 'quick', ['scripts/core_buoyancy_profile.py'], ['results/core_buoyancy_profile.txt'], None,
     'Sect. 5.2: N^2 of PREM\'s core'),
    ('neutral_core', 'quick', ['scripts/compare_neutral_core.py'], ['results/compare_neutral_core_K72_d1.txt'], None,
     'Sect. 5.3: neutral against PREM core'),
    ('nonhydro_diurnal', 'full', ['scripts/nonhydro_diurnal.py'], ['results/nonhydro_diurnal.txt'], None,
     'Table 4, Fig. 9: FCN and Love numbers of the non-hydrostatic states, routes A and B'),
    ('nonhydro_anelastic', 'full', ['scripts/nonhydro_anelastic.py'], ['results/nonhydro_anelastic.txt'], None,
     'Table 4: anelastic non-hydrostatic Earth (437.9, 431.4 days)'),
    ('nonhydro_support', 'full', ['scripts/nonhydro_support.py'], ['results/nonhydro_support.txt'], None,
     'S10, Table S11: elastic and viscous support'),
    ('depth_sensitivity', 'full', ['scripts/depth_sensitivity.py'], ['results/depth_sensitivity.txt'], None,
     'depth sensitivity of the Love numbers'),
    ('compare_ddw1999', 'full', ['scripts/compare_ddw1999.py'], ['results/ddw1999_PREMs.txt'], None,
     'S4: M2 agreement with Dehant et al. (1999)'),
    ('fcn_share_inviscid', 'full', ['scripts/fcn_share_inviscid.py'], ['results/fcn_share_inviscid.txt'], None,
     'Sect. 5.3: inviscid resonance shares (+0.0040)'),
    ('ddw_lowK', 'full', ['scripts/ddw_lowK_table.py', '3'], ['results/ddw_lowK_table.txt'], None,
     'S11: truncation, DDW comparison at low K'),
    ('conv_deflated', 'full', ['scripts/convergence_deflated.py'], ['results/convergence_deflated.txt'], None,
     'S9: solver spread after deflation'),
    ('conv_inviscid_lowK', 'full', ['scripts/convergence_inviscid_lowK.py', '4', '60', '2'],
     ['results/convergence_inviscid_K4_N60.txt'], None, 'S11: radial resolution of the inviscid core'),
    ('conv_nocoriolis', 'full', ['scripts/convergence_nocoriolis_core.py'], ['results/convergence_nocor_core.txt'],
     None, 'S9: no Coriolis force in the core, solvers agree to 1e-9'),
    ('ekman_resolution', 'full', ['scripts/check_ekman_resolution.py'], ['results/check_ekman_resolution.txt'], None,
     'S9: Ekman sub-layers'),
    ('m2_rays_recheck', 'full', ['scripts/m2_rays_recheck.py'], ['results/m2_rays_recheck.txt'], None,
     'Sect. 5.2: attractors at M2 and 0.3 per cent above (140, 9 reflections; 9 min)'),
    ('rays_m2_long', 'full', ['scripts/rays_m2_long.py'], ['results/rays_m2_long.txt'], None,
     'Sect. 5.2: no attractor at M2, 50000 reflections (83 min)'),
    ('fcn_noic_viscous', 'heavy', ['scripts/fcn_resonance_noic.py', '12', '6'],
     ['results/fcn_resonance_noic_K12.txt'], None, 'Sect. 5.3: viscous core without inner core (+0.0015)'),
    ('fcn_noic_inviscid', 'heavy', ['scripts/fcn_share_inviscid_noic.py', '6'], ['results/fcn_share_inviscid_noic.txt'],
     None, 'Sect. 5.3: inviscid core without inner core (+0.0040)'),
]

# ---------------------------------------------------------------------------------------- P1 v3 (PEPI, submitted)
# Bos, M. S., Body tides of a flattened, rotating, non-hydrostatic Earth: a transparent numerical model (PEPI;
# main_v3.tex, supplement_v3.tex). The paper uses the level figure (EARTHTIDES_FIGURE=level, or --figure level)
# except where it compares with first-order theories (Smith 1974, Dehant et al. 1999, the exact spheroid, the
# Hough and bump tests, the inviscid-shell tests): those jobs are taken from JOBS_V2 unchanged. A seventh element
# of a job is its environment.
_V2 = {j[0]: j for j in JOBS_V2}
_LEVEL = {'EARTHTIDES_FIGURE': 'level', 'VECLIB_MAXIMUM_THREADS': '1'}
_FIG3 = lambda *names: [f'figure:paper_pepi/figures/{n}.pdf' for n in names]
JOBS_V3 = [_V2[n] for n in (
    'check_hard', 'check_m', 'check_bump', 'exact_spheroid', 'smith1974', 'rotating_spheroid', 'ex3', 'ex5',
    'angular_momentum', 'fcn_period', 'frame_size', 'anisotropy_budget', 'ddw_fingerprint', 'ddw_reprojection',
    'ddw_wplus', 'quadrature_nodes', 'core_sweep_offset', 'viscous_gap', 'rays_disc', 'rays_disc_layered',
    'figures_v2', 'fig_rays_buoyancy', 'fig_layered_core', 'fig_grid', 'fig_attractor3d', 'figS_core_flow_rays',
    'figS_sphere_flat', 'am_floor', 'sphere_flat_ratios', 'core_rigidity', 'ekman_profile', 'attractor_control',
    'core_buoyancy', 'neutral_core')] + [
    # --- quick: the level figure, tables and figures of v3, the review (2026-11-15)
    ('validate_level_figure', 'quick', ['scripts/validate_level_figure.py'], ['results/validate_level_figure.txt'],
     None, 'Sect. 3.4, S10: the level figure against Maclaurin and Chambat et al. (2010); angular momentum'),
    ('love_level_figure', 'quick', ['scripts/love_level_figure.py'], ['results/love_level_figure.txt'], None,
     'Sect. 4, Table S10: Love numbers on the level against the first-order figure', _LEVEL),
    ('fcn_level_figure', 'quick', ['scripts/fcn_level_figure.py'], ['results/fcn_level_figure.txt'], None,
     'Sect. 5.1, Table 1: FCN on the level figure (456.2 days)', _LEVEL),
    ('am_wall_level', 'quick', ['scripts/am_wall_level.py'], ['results/am_wall_level_K4_N24.txt'], None,
     'Sect. 4: angular momentum on the level figure', _LEVEL),
    ('slip_nonlevel_waves', 'quick', ['scripts/slip_nonlevel_waves.py', '3', '24'], ['results/slip_nonlevel_waves.txt'],
     None, 'Sect. 5.3: the slip term at the main waves'),
    ('hough_share_slipterm', 'quick', ['scripts/hough_share_test.py', '4', '--variants',
                                       'soft,bump-1,bump+1,bump+3,clairaut,cl_exacteps,ex_cleps', '--deltas', 'none',
                                       '--slip-term', '--tag', 'slipterm'],
     ['results/hough_share_test_slipterm.txt'], None, "Sect. 5.3, S5: Hough's Earth with and without the slip term"),
    ('exact_levels_share', 'quick', ['scripts/exact_levels_share_test.py', '4', '--variants', 'clairaut,exact_levels',
                                     '--deltas', 'none', '--slip-term', '--model', 'data/PREM.dat', '--tag', 'fullPREM'],
     ['results/exact_levels_share_test_slipterm_fullPREM.txt'], None,
     'S5: exact ellipses against the first-order map (7.8 m), PREM', {'VECLIB_MAXIMUM_THREADS': '1'}),
    ('exact_levels_share_noic', 'quick', ['scripts/exact_levels_share_test.py', '4', '--variants',
                                          'clairaut,exact_levels', '--deltas', 'none', '--slip-term'],
     ['results/exact_levels_share_test_slipterm.txt'], None, 'S5: the same without inner core',
     {'VECLIB_MAXIMUM_THREADS': '1'}),
    ('convention_table', 'quick', ['scripts/convention_table.py', '--figure', 'level'],
     ['results/convention_table_level.txt'], None, 'Table S1: conventions'),
    ('table_latitude', 'quick', ['paper_pepi/make_table_latitude.py', '--figure', 'level'],
     ['results/paper1_table_latitude_level.txt', 'results/regen/table_latitude_level.tex'], None,
     'Table S7: origin of the latitude dependence'),
    ('stress_split', 'quick', ['scripts/nonhydro_stress_split.py', '450', '4', '--figure', 'level'],
     ['results/nonhydro_stress_split_450m_level.txt'], None, 'Sect. 6.3, Table S11: tau and slip terms separately'),
    ('stress_split_0', 'quick', ['scripts/nonhydro_stress_split.py', '0', '3', '--figure', 'level'],
     ['results/nonhydro_stress_split_0m_level.txt'], None, 'Table S10: routes at 0 m'),
    ('stress_split_225', 'quick', ['scripts/nonhydro_stress_split.py', '225', '3', '--figure', 'level'],
     ['results/nonhydro_stress_split_225m_level.txt'], None, 'Table S10: routes at 225 m'),
    ('stress_split_500', 'quick', ['scripts/nonhydro_stress_split.py', '500', '3', '--figure', 'level'],
     ['results/nonhydro_stress_split_500m_level.txt'], None, 'Table S10: routes at 500 m'),
    ('stress_split_441', 'quick', ['scripts/nonhydro_stress_split.py', '441', '3', '--figure', 'level'],
     ['results/nonhydro_stress_split_441m_level.txt'], None, 'Sect. 6.3, S7: with the stress at 431.4 days'),
    ('stress_split_410', 'quick', ['scripts/nonhydro_stress_split.py', '410', '3', '--figure', 'level'],
     ['results/nonhydro_stress_split_410m_level.txt'], None, 'Sect. 6.3, S7: without the stress at 431.4 days'),
    ('nonhydro_support', 'quick', ['scripts/nonhydro_support.py'], ['results/nonhydro_support_level.txt'], None,
     'Table S9: three load shells fitted to the targets', _LEVEL),
    ('slip_vs_viscous', 'quick', ['scripts/slip_vs_viscous_level.py'], ['results/slip_vs_viscous_level.txt'], None,
     'Sect. 5.3, S5: slipping against viscous core on the level figure'),
    ('delta_vs_limit', 'quick', ['scripts/viscous_delta_vs_limit.py', '--thin', 'Kthin'],
     ['results/viscous_delta_vs_limit_Kthin_level.txt'], None, 'S5: a finite delta against the limit (core K 72)',
     {'VECLIB_MAXIMUM_THREADS': '1'}),
    ('delta_fit_forms', 'quick', ['scripts/delta_fit_forms.py'], ['results/delta_fit_forms.txt'], None,
     'S5: quadratic against delta^1/2, delta^2/3 in the limit delta -> 0'),
    ('l1_frame_sensitivity', 'quick', ['scripts/l1_frame_sensitivity.py'], ['results/l1_frame_sensitivity.txt'], None,
     'S3.1: diurnal l(1) against the removal of the mantle rotation'),
    ('wplus_spindown', 'quick', ['scripts/wplus_spindown.py'], ['results/wplus_spindown.txt'], None,
     'S3.1: second-order part of the diurnal w+ and l(1) of PREM'),
    ('table_recommended', 'quick', ['paper_pepi/make_table_recommended.py'], ['paper_pepi/table_recommended.tex'], None,
     'S1: recommended Love numbers, an excerpt of src/earthtides/data/love_PREM_allwaves.json'),
    ('mathews1995_factor2', 'quick', ['scripts/mathews1995_factor2.py'], ['results/mathews1995_factor2.txt'], None,
     'S3: eq. 6 of Mathews et al. 1995a against the Conventions (m = 0, 1, 2); their l(1) against ours'),
    ('widmer_qm1_test', 'quick', ['scripts/widmer_qm1_test.py'], ['results/widmer_qm1_test.txt'], None,
     'S3: Tables S5/S6 with the Q of Widmer et al. (1991, QM1): the Q model explains neither difference'),
    ('table4_spread', 'quick', ['scripts/table4_spread.py'], ['results/table4_spread_level.txt'], None,
     'S8: Table S13 against core N 16-30 and the linear solvers (review M4)'),
    ('ef_vs_cmb_fcn', 'quick', ['scripts/ef_vs_cmb_fcn.py'], ['results/ef_vs_cmb_fcn_level.txt'], None,
     'S8: the FCN follows Delta e_f, not the CMB flattening (Table S9)'),
    ('paper_numbers_v2', 'quick', ['scripts/paper_numbers_v2.py'], ['results/paper_numbers_v2.txt'], None,
     'numbers in the text not printed by another script (incl. l1mm, latmm: mm of the IERS latitude terms)'),
    ('table_nonhydro_love', 'quick', ['paper_pepi/make_table4.py', '--figure', 'level'],
     ['results/paper1_table4_level.txt', 'results/regen/table4_level.tex', 'results/regen/tableS_viscous_level.tex'],
     None, 'Table 1, Table S13 and the viscous-support table'),
    ('table_anelastic', 'quick', ['paper_pepi/make_table_anelastic.py', '--figure', 'level'],
     ['results/paper1_table_anelastic_level.txt', 'paper_pepi/table_anelastic.tex'], None,
     'Table S5: anelastic, non-hydrostatic, against DDW and IERS'),
    ('tables_viscous', 'quick', ['paper_pepi/make_tables_viscous.py'], ['results/paper1_tables_viscous_level.txt',
     'results/regen/table_ddw_viscous_level.tex', 'results/regen/table_latitude_viscous_level.tex'], None, 'Table S2 (viscous rows) and Table S7: viscous core, delta -> 0', _LEVEL),
    ('fig_fcn', 'quick', ['paper_pepi/make_fig_fcn.py', '4'], ['results/figdata_fcn_level.npz'] + _FIG3('fig_fcn'),
     None, 'Fig. 2: h(0) and k through the diurnal band, FICN', _LEVEL),
    ('figS_level', 'quick', ['paper_pepi/make_figS_level.py'], ['results/figdata_S_level.npz']
     + _FIG3('fig3_convergence', 'fig4_latitude_M2'), None, 'Supplement S10 (convergence) and S5 (latitude dependence)', _LEVEL),
    ('fig_core', 'quick', ['paper_pepi/make_fig_core.py'], _FIG3('fig_core_sweep', 'fig_core_ekman'), None,
     'Figs 6-7 (stored level-figure sweeps)', _LEVEL),
    ('nonhydro_figures', 'quick', ['paper_pepi/make_figures.py', '--replot'], ['results/fig_nonhydro_data_level.txt']
     + _FIG3('fig_nonhydro_state', 'fig_fcn_ef', 'fig_am_sweep'), None, 'Figs 8-10', _LEVEL),
    ('fig_ddw_iers', 'quick', ['paper_pepi/make_fig_ddw_iers.py'], _FIG3('figS_ddw_iers'), None,
     'Fig. S1: published Love numbers minus ours'),
    ('fig_dispersion', 'quick', ['paper_pepi/make_fig_dispersion.py'], _FIG3('figS_dispersion'), None,
     'Fig. S2: Q and the shear modulus from 1 s to the tides'),
    ('ex7', 'quick', ['src/earthtides/examples/ex7/nonhydrostatic.py'],
     ['src/earthtides/examples/ex7/nonhydrostatic.out'], None, 'example 7: the non-hydrostatic Earth (Table S4)'),
    ('ex8', 'quick', ['src/earthtides/examples/ex8/viscous_limit.py'],
     ['src/earthtides/examples/ex8/viscous_limit.out'], None, 'example 8: the viscous limit at O1'),
    ('ex9', 'quick', ['src/earthtides/examples/ex9/figures.py'], ['src/earthtides/examples/ex9/figures.out'], None,
     'example 9: level against first-order figure'),
    # --- full: everything that needs only this package, minutes to hours on a laptop
    ] + [_V2[n] for n in ('smith1974_sign', 'ex4', 'gravimetric_fcn430', 'ex6', 'b2_profile', 'b2_split',
                          'b2_wahr_h0', 'b2_ingredients_sphere', 'b2_ingredients_flat', 'b2_hough_coll',
                          'rays_hamiltonian', 'depth_sensitivity', 'compare_ddw1999', 'fcn_share_inviscid', 'ddw_lowK',
                          'conv_deflated', 'conv_inviscid_lowK', 'conv_nocoriolis', 'ekman_resolution',
                          'm2_rays_recheck', 'rays_m2_long')] + [
    ('nonhydro_diurnal', 'full', ['scripts/nonhydro_diurnal.py', '--figure', 'level'],
     ['results/nonhydro_diurnal_level.txt'], None, 'Table S11: FCN and Love numbers of the states, routes A and B'),
    ('nonhydro_am_sweep', 'full', ['scripts/nonhydro_am_sweep.py', '450', '4'],
     ['results/nonhydro_am_sweep_450m_level.txt', 'results/nonhydro_am_sweep_450m_level.npz'], None,
     'Fig. 10: angular momentum through the diurnal band (450 m)', _LEVEL),
    ('nonhydro_am_viscous', 'full', ['scripts/nonhydro_am_viscous.py', '10', '30', '1', '0.5'],
     ['results/nonhydro_am_viscous_K10_N30_level.txt'], None, 'S7: angular momentum with a viscous core', _LEVEL),
    ('sublayer_N_prem', 'full', ['scripts/sublayer_N_prem.py'], ['results/sublayer_N_prem_K12.txt'], None,
     'resolution of the Ekman sub-layers (solver.viscous_core_resolution)'),
    ] + [(f'sweep_K{K}', 'full', ['scripts/convergence_viscous_coreK.py', str(K), '30', '0', '3', '0.0005', '--figure',
                                  'level'], [f'results/convergence_viscous_coreK{K}_N30_d0_s0.0005_level.txt'], None,
          f'Fig. 6: inviscid sweeps, core K {K} (6-7 min)', {'VECLIB_MAXIMUM_THREADS': '1'}) for K in (8, 16, 24)] + [
    (f'nonhydro_fcn_viscous_{cmb}_{route}', 'full',
     ['scripts/nonhydro_fcn_viscous.py', str(cmb), '12', '30', '1', '0.75', '0.5', '0.35'],
     [f'results/nonhydro_fcn_viscous_{"hydro" if cmb < 0 else str(cmb) + "m"}'
      f'{"" if cmb < 0 else ("_A" if route == "A" else "_B_cs_static")}_K12_N30_level.txt'], None,
     'Table S10, Fig. 9: FCN with a viscous core (about 2 min)', dict(_LEVEL, ROUTE=route))
    for cmb, route in ((-1, 'B'), (0, 'A'), (0, 'B'), (225, 'A'), (225, 'B'), (450, 'A'), (450, 'B'), (500, 'A'),
                       (500, 'B'))] + [
    # --- heavy: a large machine (28 GB), hours
    ] + [_V2[n] for n in ('attractor_K168', 'attractor_M2', 'attractor_layered', 'attractor_sphere', 'attractor_scan',
                          'sweep_K168_d0.5', 'sweep_K168_d0.25', 'sweep_K96_d1_split', 'fcn_noic_viscous',
                          'fcn_noic_inviscid')] + [
    ('fcn_extrapolation', 'heavy', ['scripts/fcn_resonance_extrapolation_v3.py', '12', '3', '--figure', 'level',
                                    '--fresh'], ['results/fcn_resonance_extrapolation_v3_K12_level.txt'], None,
     'S5: FCN resonance at delta -> 0, core K 12', {'VECLIB_MAXIMUM_THREADS': '1'}),
    ('sweep_K96', 'heavy', ['scripts/convergence_viscous_coreK.py', '96', '30', '1', '3', '0.0005', '--band', '12',
                            '--figure', 'level'], ['results/convergence_viscous_coreK96_N30_d1_s0.0005_b12_level.txt'],
     None, 'Fig. 6: viscous sweep, core K 96 (2.8 h)', {'VECLIB_MAXIMUM_THREADS': '1'}),
    ('viscous_limit_Kladder', 'heavy', ['scripts/viscous_limit_thin.py', '--K', '24,48,72,96,120', '--deltas',
                                        '1000,750,500,350', '--no-fp'], ['results/viscous_limit_thin_Kladder_level.txt'],
     None, 'S5: the limit delta -> 0 against core K (5.3 h)', {'EARTHTIDES_FIGURE': 'level'}),
    ('viscous_limit_Kthin', 'heavy', ['scripts/viscous_limit_thin.py', '--K', '72,96', '--deltas',
                                      '1000,750,500,350,250,175', '--no-fp'], ['results/viscous_limit_thin_Kthin_level.txt'],
     None, 'Sect. 5.3, S5: the viscous limit, core K 72 (55 min)', {'EARTHTIDES_FIGURE': 'level'}),
]

# figure jobs that read the large core-flow maps (Zenodo archive only): skipped, with the reason, without them
CORE_FLOW_MAPS = ['results/core_flow_attractor_K168_S16N15.npz', 'results/core_flow_attractor_layered.npz',
                  'results/core_flow_attractor_scan.npz', 'results/core_flow_attractor_sphere.npz',
                  'results/core_flow_attractor_M2.npz']
_MAPS_JOBS = ('fig_rays_buoyancy', 'fig_layered_core', 'figS_core_flow_rays', 'figS_sphere_flat',
              'sphere_flat_ratios', 'attractor_control')
JOBS_V3 = [j[:4] + ('maps',) + j[5:] if j[0] in _MAPS_JOBS else j for j in JOBS_V3]

NUM = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
CPLX = re.compile(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)([+-][\d.]+(?:[eE][+-]?\d+)?)[ij]$')
TIMING = re.compile(r'\(\s*[\d.]+\s*s\)|\[[^\]]*:\s*\d+\s*s\]|\b\d+\.\d+\s*s\b|done\s+[\d.]+'
                    r'|\d+/\d+ solved')
TOL = 1e-8


def tokens(text):
    t = TIMING.sub('', text).replace(',', ' ').replace('=', ' = ').replace('/', ' / ')
    return t.translate(str.maketrans({c: ' ' for c in '\'"{}()[]'})).split()


def ulp(tok):
    m = NUM.match(tok.lstrip('+-'))
    dec = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
    exp = int(m.group(2)[1:]) if m.group(2) else 0
    return 10.0**(exp - dec)


def numbers(tok):
    t = tok.rstrip('%')
    m = CPLX.match(t)
    if m:
        return [m.group(1), m.group(2)]
    return [t] if NUM.match(t) else None


# Numbers that are round-off on every platform (|x| and |y| below ROUNDOFF) count as equal.
ROUNDOFF = 1e-12
# Lines left out of the comparison, per output file, with the reason.
IGNORE = {'hough1895.out': (re.compile(r'numerical mode'),
                            'spurious eigenvalues of the discretization; how many are found depends '
                            'on round-off (see the README of example 5)')}


# Files whose numbers vary between platforms by more than the printed precision: a function
# (reference value) -> allowed absolute difference, with the reason.
def _hough_tol(y):
    """Example 5: the FCN of Hough's Earth is the root of a nearly singular system and varies by
    ~1e-7 in frequency between BLAS builds; periods (1 / (f - 1), > 10) then vary by ~1e-4
    relative and the relative differences (< 0.5) by ~1e-4."""
    y = abs(y)
    return 1e-3 * y if y >= 10 else (3e-7 if y >= 0.5 else 2e-4)


def _numbers_tol(y):
    """paper_numbers.txt: the diagnostics below 1e-5 (convergence, round-off of the mesh tests)
    vary between platforms by up to 10 per cent of their value; the paper quotes them to one
    digit. K1, at the tilt-over mode, is the most sensitive."""
    return 0.1 * abs(y) if abs(y) < 1e-5 else 0.0


FILE_TOL = {'hough1895.out': _hough_tol, 'paper_numbers.txt': _numbers_tol}

# Jobs that solve an ill-posed problem on purpose (the inviscid core at low truncation, S11): their solution is
# fixed only up to the spread between linear solvers, which depends on the platform and the solver version. They
# run and must write their outputs, but are not compared number by number ('ill' in the report, not a failure).
ILL_POSED = {
    'ddw_lowK': 'inviscid core at K 2-6: some lines are UNDETERMINED (solver spread up to 1e-5); the stable lines '
                'reproduce, the |block-SuperLU| column and the UNDETERMINED flags do not',
    'conv_inviscid_lowK': 'inviscid core, K 4, N 60: the solution is undetermined by design (solver spread up to '
                          '4e-4 in h0); S11 quotes only its order of magnitude',
}
_FT = None


def same_number(x, y):
    fx, fy = float(x), float(y)
    if abs(fx) < ROUNDOFF and abs(fy) < ROUNDOFF:
        return True
    if _FT and abs(fx - fy) <= _FT(fy):
        return True
    if abs(fx - fy) <= 1.01 * max(ulp(x.lstrip('+-')), ulp(y.lstrip('+-'))):
        return True
    return abs(fx - fy) <= TOL * max(abs(fy), 1.0)


def compare_text(a, b, name=''):
    global _FT
    _FT = FILE_TOL.get(name)
    if name in IGNORE:
        a, b = ('\n'.join(l for l in t.splitlines() if not IGNORE[name][0].search(l)) for t in (a, b))
    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb):
        return [f'different length ({len(ta)} vs {len(tb)} tokens)']
    out = []
    for x, y in zip(ta, tb):
        if x == y:
            continue
        nx, ny = numbers(x), numbers(y)
        if nx and ny and len(nx) == len(ny) and all(same_number(u, v) for u, v in zip(nx, ny)):
            continue
        out.append(f'{y} -> {x}')
    return out


# Arrays of the figure data compared with an absolute tolerance (convergence errors at round-off)
# or only away from the equator (l_N divides by dY/dphi, which vanishes there; the figure masks it).
# figdata_fcn: the sweep crosses the tilt-over mode and the free core nutation, where the solution
# varies by ~1e-7 between platforms; f2_err: round-off floor of the distorted-mesh test (~1e-8).
NPZ_ATOL = {'f2_err': 1e-7, 'f3_err': 1e-10, 'h0': 1e-7, 'k0': 1e-7, 'l1': 1e-7}


def compare_file(new, ref):
    if ref.suffix == '.npz':
        import numpy as np
        a, b = np.load(new), np.load(ref)
        bad = []
        for k in b.files:
            if k not in a.files or a[k].shape != b[k].shape:
                bad.append(k)
                continue
            x, y = a[k], b[k]
            if y.dtype.kind not in 'fciu':
                if not np.array_equal(x, y):
                    bad.append(k)
                continue
            if k.startswith('f4_lN') and 'f4_th' in b.files:
                keep = np.abs(90 - np.degrees(b['f4_th'])) >= 4
                x, y = x[keep], y[keep]
            atol = next((v for p_, v in NPZ_ATOL.items() if k.startswith(p_)), 1e-12)
            if not np.allclose(x, y, rtol=1e-6, atol=atol):
                bad.append(k)
        return [f'array {k} differs' for k in bad]
    if ref.suffix == '.json':
        return compare_text(new.read_text().replace('[', ' ').replace(']', ' '),
                            ref.read_text().replace('[', ' ').replace(']', ' '))
    return compare_text(new.read_text(), ref.read_text(), ref.name)


def wpath(o, installed):
    """Path of a job file in the work tree (the examples move to examples/ for --installed)."""
    return o.replace('src/earthtides/examples/', 'examples/', 1) if installed else o


def copy_tree(dest, installed=False):
    keep = ('scripts', 'data', 'results', 'paper', 'paper_pepi', 'tools', 'pyproject.toml') + \
        (() if installed else ('src',))
    # the examples at the top as well (scripts load examples/ex5; installed: the examples only, not the package)
    ex = ROOT / 'examples' if (ROOT / 'examples').is_dir() else ROOT / 'src' / 'earthtides' / 'examples'
    shutil.copytree(ex, dest / 'examples', ignore=shutil.ignore_patterns('__pycache__'))
    (dest / 'docs' / 'manual').mkdir(parents=True, exist_ok=True)   # figure helpers of the manual
    if (ROOT / 'docs' / 'manual' / 'make_figures.py').exists():
        shutil.copy(ROOT / 'docs' / 'manual' / 'make_figures.py', dest / 'docs' / 'manual')
    for k in keep:
        s = ROOT / k
        if s.is_dir():
            shutil.copytree(s, dest / k, ignore=shutil.ignore_patterns('__pycache__', '*.pdf',
                                                                       '*.aux', '*.log', '*.bbl'))
        elif s.exists():
            shutil.copy(s, dest / k)
    (dest / 'paper' / 'figures').mkdir(parents=True, exist_ok=True)
    (dest / 'paper_pepi' / 'figures').mkdir(parents=True, exist_ok=True)


def requirement_ok(req, a):
    if req is None:
        return True, ''
    if req == 'maps':
        miss = [m for m in CORE_FLOW_MAPS if not (ROOT / m).exists()]
        return not miss, ('core-flow maps not found (they are in the Zenodo archive, not in the git '
                          'repository): ' + ', '.join(Path(m).name for m in miss))
    if req == 'sympy':
        try:
            import sympy  # noqa: F401
            return True, ''
        except ImportError:
            return False, 'sympy not installed'
    if req == 'grids':
        g = Path(a.grids)
        need = [g / m / 'amp.k1' for m in ('FES2014b', 'FES2022b', 'GOT5.5', 'TPXO10_atlas_v2')]
        miss = [str(p.parent) for p in need if not p.exists()]
        return not miss, 'ocean tide grids missing: ' + ', '.join(miss)
    if req == 'fes2004':
        ok = a.iers and (Path(a.iers) / 'fes2004.dat').exists()
        return bool(ok), 'fes2004.dat not found (--iers DIR)'
    if req == 'iers':
        ok = a.iers and (Path(a.iers) / 'de421.bsp').exists() and (Path(a.iers) / 'dehant').exists()
        try:
            import skyfield  # noqa: F401
        except ImportError:
            return False, 'skyfield not installed'
        return bool(ok), 'DE421 and the DEHANTTIDEINEL variants not found (--iers DIR)'
    return False, req


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--quick', action='store_const', dest='level', const='quick')
    g.add_argument('--full', action='store_const', dest='level', const='full')
    g.add_argument('--external', action='store_const', dest='level', const='external')
    g.add_argument('--heavy', action='store_const', dest='level', const='heavy',
                   help='v2: also the jobs that need a large machine (hours, up to 24 GB)')
    p.add_argument('--report-only', action='store_true',
                   help='differences (DIFF) are reported but do not fail the run; only scripts that crash do '
                        '(continuous integration on platforms other than the reference machine)')
    p.add_argument('--paper', choices=('v1', 'v2', 'v3'), default='v3',
                   help='v3: the PEPI paper as submitted (main_v3, supplement_v3; default); v2: its earlier '
                        'version; v1: the J Geod paper it grew from')
    p.add_argument('--list', action='store_true')
    p.add_argument('--only', nargs='+')
    p.add_argument('--keep')
    p.add_argument('--update', action='store_true',
                   help='write the new outputs over the stored results (after a code change)')
    p.add_argument('--installed', action='store_true',
                   help='use the installed earthtides package (pip install earth-tides), not src/')
    p.add_argument('--grids', default=str(Path.home() / 'projects/otl_map/surfaceloads'))
    p.add_argument('--iers', default=str(Path.home() / 'scratch/earthtides_iers'))
    a = p.parse_args()
    level = a.level or 'quick'
    order = {'quick': 0, 'full': 1, 'external': 2, 'heavy': 3}
    J = {'v1': JOBS, 'v2': JOBS_V2, 'v3': JOBS_V3}[a.paper]
    if a.list:
        for name, lev, cmd, outs, req, what, *rest in J:
            print(f'{name:22s} {lev:9s} {what}')
        return
    jobs = [j for j in J if (a.only and j[0] in a.only) or (not a.only and order[j[1]] <= order[level])]
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix='earthtides_repro_'))
    if work.exists() and any(work.iterdir()):
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    copy_tree(work, a.installed)
    env = dict(os.environ, PYTHONPATH='' if a.installed else str(work / 'src'), PYTHONWARNINGS='ignore',
               MPLBACKEND='Agg')
    r = subprocess.run([PY, '-c', 'import earthtides; print(earthtides.__version__, earthtides.__file__)'],
                       cwd=work, env=env, capture_output=True, text=True)
    print(f'earthtides {r.stdout.strip() or r.stderr.strip().splitlines()[-1]}', flush=True)
    (work / 'results' / 'stdout').mkdir(exist_ok=True)
    report, t_all = [], time.time()
    try:
        for name, lev, cmd, outs, req, what, *rest in jobs:
            jenv = dict(env, **(rest[0] if rest else {}))      # per-job environment, e.g. EARTHTIDES_FIGURE
            ok, why = requirement_ok(req, a)
            if not ok:
                report.append((name, 'skip', why))
                print(f'[skip] {name}: {why}', flush=True)
                continue
            for o in outs:                                # the stored output must not be read back
                if not o.startswith(('stdout:', 'figure')) and (work / wpath(o, a.installed)).exists():
                    (work / wpath(o, a.installed)).unlink()
            args = [PY] + [c.replace('{IERS}', str(a.iers)) for c in cmd]
            cwd = work
            if cmd[0].startswith('src/earthtides/examples/'):
                cwd = work / Path(wpath(cmd[0], a.installed)).parent
                args = [PY, Path(cmd[0]).name]
            t = time.time()
            r = subprocess.run(args, cwd=cwd, env=jenv, capture_output=True, text=True)
            dt = time.time() - t
            if r.returncode:
                last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f'exit {r.returncode}'
                report.append((name, 'FAIL', last))
                print(f'[FAIL] {name} ({dt:.0f} s): {last}', flush=True)
                continue
            diffs = []
            for o in outs:
                if o == 'figures':
                    n = len(list((work / 'paper' / 'figures').glob('*.pdf')))
                    diffs += [] if n else ['no figures written']
                    continue
                if o.startswith('figure:'):                  # PDFs are not copied: it must be new
                    diffs += [] if (work / o[7:]).exists() else [f'{o[7:]} not written']
                    continue
                if o.startswith('stdout:'):
                    new = work / 'results' / 'stdout' / (o[7:] + '.txt')
                    new.write_text(r.stdout)
                    ref = ROOT / 'results' / 'stdout' / (o[7:] + '.txt')
                else:
                    new, ref = work / wpath(o, a.installed), ROOT / o
                    if not ref.exists():                 # examples also at the top of the release
                        ref = ROOT / wpath(o, True)
                if name == 'love_table':
                    ref = ROOT / 'src' / 'earthtides' / 'data' / 'love_PREM.json'
                    if a.installed or not ref.exists():  # the table shipped with the package
                        q = subprocess.run([PY, '-c', 'import earthtides, os; print(os.path.dirname('
                                            'earthtides.__file__))'], cwd=work, env=env,
                                           capture_output=True, text=True)
                        ref = Path(q.stdout.strip()) / 'data' / 'love_PREM.json'
                if not ref.exists():
                    diffs.append(f'no stored reference {ref.relative_to(ROOT)}')
                    continue
                if not new.exists():
                    diffs.append(f'{o} not written')
                    continue
                d = compare_file(new, ref) if name != 'love_table' else compare_love(new, ref)
                diffs += [f'{Path(o).name}: {x}' for x in d[:3]]
            if a.update:                                 # new outputs become the stored results
                for o in outs:
                    if o == 'figures':
                        for f in (work / 'paper' / 'figures').glob('*.pdf'):
                            shutil.copy(f, ROOT / 'paper' / 'figures' / f.name)
                        continue
                    if o.startswith('figure:'):
                        src, dst = work / o[7:], ROOT / o[7:]
                    elif o.startswith('stdout:'):
                        src, dst = work / 'results' / 'stdout' / (o[7:] + '.txt'), ROOT / 'results' / 'stdout' / (o[7:] + '.txt')
                    else:
                        src, dst = work / wpath(o, a.installed), ROOT / o
                    if name == 'love_table':
                        dst = ROOT / 'src' / 'earthtides' / 'data' / 'love_PREM.json'
                    if src.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src, dst)
            status = 'ok' if not diffs else 'DIFF'
            if name in ILL_POSED and not any('not written' in x or 'no stored' in x for x in diffs):
                status, diffs = 'ill', [f'ill-posed, not compared ({ILL_POSED[name]})']
            report.append((name, status, '; '.join(diffs)))
            print(f'[{status:4s}] {name} ({dt:.0f} s){": " + "; ".join(diffs) if diffs else ""}', flush=True)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    n_ok = sum(1 for r in report if r[1] == 'ok')
    print(f'\n{n_ok} of {len(report)} jobs reproduce the stored results '
          f'({sum(1 for r in report if r[1] == "skip")} skipped, '
          f'{sum(1 for r in report if r[1] == "ill")} ill-posed by design), {time.time() - t_all:.0f} s')
    if a.keep:
        print(f'outputs kept in {work}')
    accept = ('ok', 'skip', 'ill') + (('DIFF',) if a.report_only else ())
    if a.report_only and any(r[1] == 'DIFF' for r in report):
        print('--report-only: differences from the stored results are reported, not fatal (other platforms and '
              'BLAS libraries change the last digits; the exact reproduction is checked on the reference machine)')
    sys.exit(0 if all(r[1] in accept for r in report) else 1)


def compare_love(new, ref):
    """Love tables: the entries only (the metadata hold dates and runtimes), to 1e-6 relative: the
    K1 entry, at the frequency of the tilt-over mode, differs by ~3e-7 between NumPy builds."""
    import json
    global TOL
    a, b = json.loads(new.read_text()), json.loads(ref.read_text())
    tol, TOL = TOL, 1e-6
    try:
        return compare_text(json.dumps(a['entries']).replace('[', ' ').replace(']', ' '),
                            json.dumps(b['entries']).replace('[', ' ').replace(']', ' '))
    finally:
        TOL = tol


if __name__ == '__main__':
    main()
