## Example 8: the viscous core and the limit of vanishing viscosity

An inviscid liquid core inside a shell with an inner core has an ill-posed forced response: a resolved
numerical solution depends on the resolution. A viscous core is well posed, and its limit of vanishing
viscosity is the reference (manual, Sect. "The viscous core").

### What the script does

For O1 and elastic PREM on the level figure, the core gets a viscosity with an Ekman-layer thickness
δ = 1000, 500 and 250 m (`EarthModel.viscous_core`: a purely imaginary shear modulus μ = iωη, welded
boundaries, sub-layers of 40 km and 5δ at every wall with their own Chebyshev nodes), with K = 12 in
the core. The Love numbers are extrapolated to δ → 0 with a quadratic in δ and compared with the coarse
inviscid core that slips along its boundaries (K = 4, N = 24).

### What to look for

The Ekman layers change h0 at O1 by about 5e-6 per kilometre of δ, and the limit agrees with the
slipping core to a few 1e-6: away from the free core nutation the choice of core hardly
matters. Next to the resonance it does, and there the paper fits the resonance as a complex pole at each
δ (core K = 72), which is beyond a laptop example.

    python viscous_limit.py         (about 30 seconds, 1 GB; compare with viscous_limit.out)
