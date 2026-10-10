# Reproducing the paper

Scripts and stored results for

> Bos, M. S., *What the IERS body-tide model assumes: Love numbers of a flattened, rotating,
> anelastic Earth*, Journal of Geodesy (submitted), and its Online Resource 1.

Everything in the paper was computed with the Python package `earthtides` in `src/` (Python ≥ 3.10,
NumPy, SciPy; Matplotlib for the figures). The scripts that produce the tables, figures and numbers
are in `scripts/`, the figure script in `paper/make_figures.py`, and their outputs, as used in the
paper, in `results/`.

## Running

```
pip install numpy scipy matplotlib sympy  # netCDF4 and skyfield only for --external
python paper_scripts/run_all.py --list    # the jobs and what they produce
python paper_scripts/run_all.py           # quick: ~3 minutes
python paper_scripts/run_all.py --full    # everything that needs only this package: ~1.5 hours
python paper_scripts/run_all.py --external --grids DIR --iers DIR
```

Each job runs in a temporary copy of this directory, so `results/` is never overwritten; its
outputs are compared with the stored ones number by number (a printed number may differ by one unit
in its last digit) and reported as `ok` or `DIFF`. `--keep DIR` keeps the new outputs.

**External inputs** (`--external`), not distributed here:

| input | used for | obtain from |
|---|---|---|
| ocean tide grids FES2014b, FES2022b (amp./pha. per wave, GMT netCDF) | Tables 5-7, Fig. 5 | AVISO, https://www.aviso.altimetry.fr |
| GOT5.5 | Table 7, Fig. 5 | NASA GSFC, https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models |
| TPXO10-atlas v2 | Table 7, Fig. 5 | https://www.tpxo.net |
| DEHANTTIDEINEL (IERS Conventions, chapter 7, revision 2016-12-19), compiled in variants by `tools/iers/make_variants.py` | Sect. 7, Table S1 | https://iers-conventions.obspm.fr |
| fes2004.dat, the FES2004 coefficients of the IERS Conventions (Sect. 6.3) | Table 5 | https://iers-conventions.obspm.fr |
| JPL ephemeris DE421 (`de421.bsp`) and the Python package skyfield | Sect. 7, Table S1 | https://ssd.jpl.nasa.gov/planets/eph_export.html |

The ocean tide loading at the superconducting-gravimeter stations
(`results/tidal_gravity/CARGA_*.csv`, Sect. 6.4) was computed with the program CARGA, which is not
distributed; these files are inputs of the gravity jobs. The control files are stored next to them.

## Where each result comes from

| paper | job (`run_all.py`) | script | stored output |
|---|---|---|---|
| Table 1, verification | `check_hard`, `check_m`, `check_bump`, `exact_spheroid`, `ex3`, `ex4`, `ex5`, `angular_momentum`, `paper_numbers` (mesh, FCN spin-up) | `scripts/check_*.py`, `src/earthtides/examples/ex3-ex5` | `results/stdout/`, `results/exact_spheroid.txt`, `results/paper_numbers.txt` |
| Table 2, comparison with Dehant et al. (1999); Fig. 4 | `diurnal` | `scripts/run_diurnal.py` | `results/diurnal_PREM.txt`, `results/figdata_fcn.npz` |
| Table 3, anelastic increase; Sect. 5.3 | `anelastic`, `compare_models`, `displacement_budget` (PREM at 200 s) | `scripts/run_anelastic.py`, `scripts/compare_models.py` | `results/anelastic_PREMs.txt`, `results/compare_models.txt`, `results/love_PREM_T0_200s.json` |
| Sect. 5.2, Coriolis part of l(1) | `coriolis`, `paper_numbers` | `scripts/run_coriolis.py` | `results/coriolis_PREMs.txt` |
| Sect. 6.1, FCN period, Hough's formula, compliance | `fcn_period`, `ex5`, `paper_numbers` | `scripts/fcn_period.py`, `scripts/paper_numbers.py` | `results/paper_numbers.txt` |
| Table 4, FCN residues (this paper) | `paper_numbers` | `scripts/paper_numbers.py` | `results/paper_numbers.txt` |
| Sects 6.2-6.3, Tables 5-6, load resonance and ocean admittance | `otl_fcn_resonance`, `iers_admittance`, `check_chao1996` | `scripts/otl_fcn_resonance.py`, `scripts/check_iers_admittance.py`, `scripts/check_chao1996.py` | `results/otl_fcn_resonance.txt` |
| Sect. 6.4, tidal gravity | `scan_delta_plus`, `gravimetric_spinup`, `gravimetric_fcn430`, `k1o1_sensitivity`, `ducarme`, `tidal_gravity_k1` | `scripts/gravimetric_factor_*.py`, `scripts/check_tidal_gravity_*.py` | `results/gravimetric_factor_*.txt`, `results/stdout/scan_delta_plus.txt`, `results/tidal_gravity/` |
| Sect. 7, Love table of PREM | `love_table` | `earthtides-love` (`src/earthtides/love.py`) | `src/earthtides/data/love_PREM.json` |
| Sect. 7.1, PREM - IERS per station and line | `prem_iers_harmonics` | `scripts/prem_iers_harmonics.py` | `results/prem_iers_harmonics.txt` |
| Sect. 7.2, Fig. 5, Table 7, uncertainty at nine stations | `station_error_budget` | `scripts/station_error_budget.py` | `results/station_error_budget.txt`, `.npz` |
| Sect. 8.3, Table S3, conventions | `convention_table` | `scripts/convention_table.py` | `results/convention_table.txt` |
| Table S1, etide vs the IERS routine (S6) | `etide_vs_iers`, `body_tide_check` (after `station_admittance`) | `scripts/etide_vs_iers.py`, `scripts/body_tide_check.py` | `results/etide_vs_iers.txt`, `results/body_tide_check.txt` |
| Table S2, Fig. S3, budget per station and line (S7) | `displacement_budget`, `figures` | `scripts/displacement_budget.py`, `paper/make_figures.py` | `results/displacement_budget.txt` |
| Figs 1-5, S1-S3 | `figures` (stored data), `figure_data` (recomputed) | `paper/make_figures.py` | `paper/figures/`, `results/figdata*.npz` |
| other numbers in the text | `paper_numbers` | `scripts/paper_numbers.py` | `results/paper_numbers.txt` |

`results/paper_numbers.txt` lists each number with the value printed in the paper next to it.

**Not reproduced here:** the change of h2 for the transversely isotropic version of PREM (Sect. 2.4)
was computed with the anisotropic code of Bos & Scherneck (2013). Values taken from the literature
(the IERS column of Table 3, the Sasao & Wahr and IERS columns of Table 4, the observed FCN periods,
the observed gravimetric factors) are quoted, not computed.
