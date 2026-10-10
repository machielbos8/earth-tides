# earth-tides

earth-tides computes solid Earth tides: the Love numbers of a rotating, flattened, layered,
anelastic and self-gravitating Earth, and the tidal displacement of a station that follows from
them. The Love numbers are computed directly on the flattened Earth by a spectral method
(Chebyshev collocation in radius, Galerkin projection along the level surfaces), without
generalized spherical harmonics or hand-derived perturbation theory. The package is pure Python
(NumPy and SciPy).

## Quick start

earth-tides has no graphical interface and no icon to click: everything happens in a terminal.
Open one (Windows: *Command Prompt* or *PowerShell*; macOS: *Terminal*; Linux: any shell), then
**type each command below and press Enter**, waiting for one to finish before starting the next.

**Step 1: install earth-tides** (requires Python 3.10 or newer with `pip`):

```bash
pip install earth-tides
```

Nothing works until this step has completed successfully.

**Step 2: get the examples and the manual:**

```bash
earthtides-examples
```

This copies five worked examples and the PDF user manual into a new directory
`earthtides-examples` in your current location (give a different name as argument if you prefer,
e.g. `earthtides-examples my_dir`). If the command is not found, step 1 did not finish
successfully: read its error messages.

Open `earthtides_manual_v1.0.pdf` first. It explains the conventions of tidal Love numbers, how
the solver works, the two programs and every example.

**Step 3: compute something.** The body tide at Onsala for one day, hourly, with the Love numbers
of the IERS Conventions (columns: MJD, up, north, east in metres):

```bash
etide 57.395 11.926 --start 2026-01-01 --days 1
```

The Love numbers of PREM for every wave group of the tidal catalogue (about 12 seconds), and the
same body tide computed with them:

```bash
earthtides-love -o love.json
etide 57.395 11.926 --start 2026-01-01 --days 1 --love love.json
```

**Step 4: run an example.** Each example runs from its own directory and prints the numbers of
its `.out` file:

```bash
cd earthtides-examples/ex1
python swing.py
```

| example | topic | runtime |
|:---|:---|---:|
| ex1 | The pumped swing: trial function, residual and projection in one variable | 1 s |
| ex2 | The loaded elliptical drum: collocation and projection on level curves | 2 s |
| ex3 | Homogeneous spheres: exact Bessel solution (Bos & Scherneck 2013) | 2 s |
| ex4 | Love numbers of a flattened homogeneous Earth (Greff-Lefftz et al. 2005) | 6 min |
| ex5 | Free core nutation of a rigid shell with a fluid core (Hough 1895) | 20 s |

## Programs

| Name | Description |
|:--- |:--- |
| `etide` | Body-tide displacement of a station (up, north, east) from the Tamura (1987) tidal potential, with the IERS Love numbers or a table of `earthtides-love` |
| `earthtides-love` | Love numbers per wave group (default) or per wave, every convention as an option, written to a JSON table |
| `earthtides-examples` | Copy the examples and the manual to a new directory |

All print their options with `--help`.

## From Python

```python
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

md = EarthModel(str(model_path('PREM')))   # PREM with a fluid outer core
g = ClairautGeometry(md)                     # hydrostatic flattening (Clairaut)
S = TidalSolver(md, g, m=2, K=5, N=30, omega_f=0.0, coriolis=OMEGA_EARTH,
                omega_t=1.4053e-4, omega_rot=OMEGA_EARTH,
                dyn_fluid=True, consistent_reference=True)   # M2
S.solve()
print(iers_fit(S, g))     # h0 = 0.60243, h2 = -0.00049, l0 = 0.08361, l1 = 0.00099, ...
```

## Verification

Every part of the method is checked against a known answer: homogeneous spheres (exact Bessel
solution), a spherical Earth on a distorted mesh (round-off), an exact homogeneous spheroid
(round-off), the analytical flattened Earth of Greff-Lefftz et al. (2005) (5e-6), Hough's free
core nutation (2e-4), and elastic hydrostatic PREM against Dehant, Defraigne & Wahr (1999)
(1e-5 at M2). `etide` reproduces the IERS routine DEHANTTIDEINEL to 0.1 mm rms.

## Reference

If you use earth-tides in your research, please cite:

> Bos, M.S. (2026). What the IERS body-tide model assumes: Love numbers of a flattened, rotating,
> anelastic Earth. *Journal of Geodesy* (submitted).

## License

Free for academic, research, and educational use. Commercial use requires a separate license from
[TeroMovigo – Earth Innovation Lda](https://teromovigo.com). See LICENSE for the full terms.
