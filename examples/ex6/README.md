## Example 6: Ekman damping of the free core nutation

Example 5 checks the free core nutation of Hough's (1895) Earth with an inviscid
core, which slides freely along the core–mantle boundary. This example gives the
same core a viscosity, so that it sticks to the boundary (no slip), and checks the
solver against the classical Ekman-layer result (paper, Sect. 7).

### The physics

A viscous fluid that moves relative to a wall is brought to rest in a thin
boundary layer, the Ekman layer, of thickness δ = (ν/Ω)^(1/2), ν = η/ρ the
kinematic viscosity. The stress in this layer exerts a torque on the core that
opposes its rotation relative to the mantle. In the theory of the nutations
(Sasao et al. 1980; Mathews et al. 2002) such a torque enters as a complex
addition to the dynamical ellipticity of the core, e_f → e_f + K, so that for a
rigid mantle

    omega_FCN = Omega [1 + (A / A_m) (e_f + K)],   K = (c_r + i c_i) E^(1/2),

with E = ν/(Ω r_c²) the Ekman number, so E^(1/2) = δ/r_c. The imaginary part
damps the mode (time dependence exp(iωt)). The coefficients follow from the
Ekman-layer stress integrated over a spherical wall (`ekman_coefficient` in
`ekman.py`, which shares no code with the solver): c_r + i c_i = 0.2585 + 2.6205 i
for a mode at one cycle per day in the frame of the wall, the spin-over correction
of Greenspan (1968). It depends on the frequency through the critical latitudes,
where the local Coriolis frequency 2Ω cos θ equals the frequency of the mode and
the Ekman layer thickens.

### The model

Hough's Earth of example 5 (`HOUGH.dat`, written by `../ex5/hough1895.py`):
homogeneous, ρ = 5500 kg/m³, Maclaurin spheroid, nearly incompressible fluid core
of radius 3480 km, mantle 10⁴ times stiffer than PREM. `EarthModel.viscous_core`
gives the core the shear modulus μ = iωη of a Newtonian fluid (correspondence
principle) and cuts it at 40 km and 5δ below the core–mantle boundary, so that
the Ekman layer has its own Chebyshev nodes.

### How the mode is found

The free mode is the complex frequency at which the solver's system matrix H(ω)
is singular. Muller's method finds the root of det H, with the rows of H scaled
once, at the starting frequency, so that det H stays an analytic function of ω
(for complex ω, μ = iωη is the analytic continuation). The inviscid root (example 5)
is the reference; the solver's shift (root − inviscid) / ((A/A_m) δ/r_c) is
compared with c_r + i c_i at the frequency of the root.

### Run

    python ekman.py

About 10 minutes on 6 cores; writes `ekman.out`.

### Output

See `ekman.out`. With K = 12 the solver's correction is 0.346 + 2.654 i at
δ = 4 km and 0.280 + 2.640 i at δ = 0.25 km, against 0.244 + 2.618 i: the damping
agrees to 0.8 per cent and the whole correction to 1.6 per cent at the smallest δ.
The difference falls slowly as δ decreases (roughly as δ^0.2–0.4): the structure
near the critical latitudes (30° latitude), where the stress has a square-root
cusp, is not resolved by K ≤ 16. At δ = 1 km the K = 6 … 16 results scatter by
±0.03 around 0.29 + 2.64 i for the same reason.

The test suite (`tests/test_viscous.py`) runs the δ = 1 km, K = 6 case (4 per cent).
