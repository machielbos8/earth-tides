## Example 5: The free core nutation of Hough's Earth

Hough (1895) found the free core nutation for a rigid ellipsoidal shell that
contains a homogeneous, incompressible fluid. This example builds that Earth
and checks that the solver finds the mode at the classical frequency (paper,
Sect. 5.2).

### The physics

Because the core–mantle boundary is flattened, the fluid core can rotate about
an axis slightly inclined to that of the mantle, coupled to it by the pressure of
the fluid on the boundary. In a spherical cavity the core could rotate about any
axis without exerting a torque, and the mode would be at exactly one cycle per
sidereal day in the rotating frame. In a flattened cavity the fluid must follow
the boundary: in space the axis of the core precesses slowly and retrograde, at a
rate proportional to the flattening, while the mantle moves the opposite way so
that the total angular momentum is conserved. For a rigid mantle the frequency in
the rotating frame is (Hough 1895; modern form in Rekier 2022, *Planet. Sci. J.* 3, 133)

    omega_FCN = Omega [1 + (A / A_m) e_f]

with e_f = (C_f − A_f)/A_f the dynamical ellipticity of the core and A, A_m the
equatorial moments of inertia of the whole Earth and of the mantle. The period in
space is 1/[(A/A_m) e_f] sidereal days.

### The model

`hough1895.py` writes `HOUGH.dat` (format as in `data/`):

* homogeneous, self-gravitating body, ρ = 5500 kg/m³, radius 6371 km;
* the exact equilibrium shape for its rotation rate, a Maclaurin spheroid
  (flattening 1/231 for the Earth's rotation rate); all layers are similar
  spheroids;
* fluid core of radius 3480 km (Vp = 1000 km/s: nearly incompressible);
* mantle 10⁴ times stiffer than that of PREM (Vs = 1000 km/s): rigid.

With equal densities there is no gravitational torque between mantle and core,
and for similar homogeneous layers A_m/A = 1 − (r_c/a)⁵ and
e_f = (a² − c²)/(a² + c²), so the prediction has no free parameters.

### How the mode is found

The solver is run at tidal frequencies near the prediction, with Coriolis and
inertial forces in both layers and free rotations (`wahr_frame=True`,
`dyn_fluid=True`), and with the reference potential computed from the model's
own density (`consistent_reference=True`). The free mode is the frequency at
which the linear system of the solver becomes singular: its determinant changes
sign there, and the root is found with Brent's method. (Fitting a resonance curve
to a forced response is less reliable: the tilt-over mode at exactly one cycle
per sidereal day lies close by.)

### Run

    python hough1895.py

About 20 seconds; writes `hough1895.out`.

### Output

```
   Omega      1/f  K    predicted       solver   P pred   P solv  rel diff
   2.0xE    57.60  6   1.01840613   1.01838988    54.33    54.38  -8.8e-04   FCN
   1.0xE   231.07  6   1.00455866   1.00455771   219.36   219.41  -2.1e-04   FCN
   1.0xE   231.07  8   1.00455866   1.00455759   219.36   219.41  -2.3e-04   FCN
   0.5xE   924.94  6   1.00113702   1.00113720   879.49   879.35  +1.6e-04   FCN
without reference potential (no centrifugal pre-stress), Omega_E: -2.5e-02
```

(frequencies in cycles per sidereal day in the rotating frame, P = period in
space in sidereal days; rotation rate as a multiple of the Earth's, 1.0xE.)

* For the Earth's rotation rate the solver finds a period of 219.41 sidereal
  days, against 219.36 from Hough's formula: a relative difference of
  2 × 10⁻⁴, the same for K = 6 and K = 8.
* For a body rotating twice as fast, four times as flattened, the difference is
  9 × 10⁻⁴: the formula is of first order in e_f, the solver is not.
* The last line repeats the Earth case with the reference potential switched
  off (`gravity=False`, which also removes the centrifugal reference state).
  The frequency is then wrong by 2.5 per cent: the free core nutation tests the
  whole reference state, not only the Coriolis force.

`hough1895.out` also lists a few other sign changes of the determinant, marked
"numerical mode", 15–25 per cent away from the FCN. They are artefacts of the
discretized compressible fluid: at the Earth's rotation rate the +17 % root
moves to +21 % when the Chebyshev order N is raised from 16 to 24, to +32 % when
the core is made three times less compressible, and disappears for
Vp = 10 000 km/s, while the FCN root stays within 3 × 10⁻⁴ of the prediction.
(A homogeneous compressible fluid under gravity is slightly unstably
stratified.)

### PREM for comparison

For PREM the formula gives, with e_f = 1/392.9 and A_m/A = 0.886, a period of
348 days; the solver with mantle and inner core stiffened 10⁴ times gives 356
days, and for elastic PREM 457 days (Dehant et al. 1999: 458). The elastic
yielding of the mantle and the boundary lengthens the period by almost 30 per cent.
