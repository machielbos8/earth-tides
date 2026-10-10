## Example 4: Analytical Love numbers of a flattened homogeneous Earth

Comparison with the analytical solution of Greff-Lefftz, Métivier & Legros
(2005, *Celest. Mech. Dyn. Astr.* 93, 113–146) for a homogeneous,
incompressible, self-gravitating elastic Earth with its hydrostatic
flattening (paper, Sect. 4.4).

### The model

One homogeneous layer, ρ = 5520 kg/m³, radius 6371 km. The shear modulus is
chosen so that the spherical Love numbers equal theirs: their constants imply
μ̄ = 19μ/(2ρga) = 3.15592, so that h₂ = (5/2)/(1 + μ̄) = 0.601553. The material
is made nearly incompressible with λ = 10⁵ μ. `greff2005.py` writes the model
file `HOMOGI.dat` (format as in `data/`). The hydrostatic flattening of this
body (Clairaut) is 1/232.1. The tide is the static semidiurnal one (m = 2).

### What is compared

On a flattened Earth the response to a degree-2 tide is no longer a pure degree 2.
Greff-Lefftz et al. express it in an ellipsoidal basis (their eqs 48–53):

| symbol | meaning |
|---|---|
| h₂ + h_o | displacement normal to the ellipsoid, degree 2 |
| h₊ | normal displacement, degree 4 |
| l₂ + l_o | tangential displacement, degree 2 |
| l₊ | tangential displacement, degree 4 |
| l* | toroidal displacement, degree 3 |
| k₂ + k_o | geoid on the deformed surface, degree 2 |
| k₊ | geoid, degree 4 |

The script projects our surface displacement and potential onto the same basis.

### Run

    python greff2005.py

About 6 minutes; writes `greff2005.out`.

### Output

```
  sphere: h2 = 0.601554 (0.601553)  l2 = 0.180466 (0.180466)  k2 = 0.360932 (0.360932)
                          no centrifugal  with centrifugal  Greff et al.
  h2+ho (normal)               +0.603145         +0.603979     +0.603140
  h+                           -0.000334         -0.000334     -0.000332
  l2+lo (tangential)           +0.181548         +0.181799     +0.181544
  l+                           -0.000064         -0.000064     -0.000064
  l* (toroidal)                -0.000226         -0.000226     -0.000224
  k2+ko (geoid)                +0.363826         +0.364328     +0.363824
  k+ (geoid)                   -0.000633         -0.000634     -0.000633
```

The first line checks the spherical limit. The column "no centrifugal" uses,
as Greff-Lefftz et al. do, a reference gravity that is purely gravitational in
the perturbation equations; it agrees with their values to 5 × 10⁻⁶. The column
"with centrifugal" also includes the centrifugal force on the displaced
parcel, as in Wahr (1981) and in the rest of the paper; it changes the degree-2
numbers by 2.5 × 10⁻⁴ (l) to 8 × 10⁻⁴ (h), and the degree-4 numbers hardly.
