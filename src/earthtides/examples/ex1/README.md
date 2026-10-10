## Example 1: The pumped swing

The method of the paper in one variable (Sect. 3.4). No Earth model is needed;
the script uses only NumPy and SciPy.

### The problem

A swing is pushed once per period while its rider pumps it standing up: she
crouches at both ends of the swing and stands up as she passes through the
bottom. Crouching lowers her centre of mass, which makes the pendulum longer and
its restoring force weaker. Her posture depends on the height of the swing, not
on its side, so it changes twice per period. With time scaled so that the
frequency of the push is one:

    x'' + w0^2 (1 + eps cos 2t) x = cos t ,     w0^2 = 2,  eps = -0.1

`cos t` is the push and `eps cos 2t` the change of posture (weakest restoring
force at the ends, t = 0 and pi). The term `eps cos 2t` plays the role of the
flattening of the Earth. Both are even: the posture is the same at both ends of
the swing, and the flattening is the same in both hemispheres,
P2(cos θ) = (1 + 3 cos 2θ)/4.

### The procedure

1. **Trial function.** `x = a1 cos t + a3 cos 3t + ...` with K odd harmonics
   (the even ones are not excited, like the degrees n with n − n_f odd).
2. **Residual.** Substituting the trial function into the equation and
   subtracting the right-hand side gives

       E(t) = sum_n a_n [w0^2 (1 + eps cos 2t) - n^2] cos nt - cos t .

   Because cos 2t cos nt = [cos(n−2)t + cos(n+2)t]/2, harmonic n contributes to
   the harmonics n−2, n and n+2 of the residual.
3. **Projection.** Require that E(t) contains none of the kept harmonics: its
   Fourier coefficients of cos t, cos 3t, ... vanish. This gives K linear
   equations for the K amplitudes. For K = 2:

       [w0^2 (1 + eps/2) - 1] a1 + (eps w0^2 / 2) a3 = 1
       (eps w0^2 / 2) a1 + (w0^2 - 9) a3             = 0

   The off-diagonal term is the coupling coefficient.

### Run

    python swing.py

Writes `swing.out` and, if matplotlib is installed, `residual.png`.

### Output

```
 K   a1               |error a1|   max |E(t)|
 1   1.1111111111     1.8e-03     1.1e-01
 2   1.1093502377     1.1e-07     1.6e-03
 3   1.1093501285     1.0e-12     6.9e-06
 4   1.1093501285     0.0e+00     1.5e-08
first-order perturbation theory: a1 = 1.1000000000, |error| = 9.4e-03
```

The reference value a1 = 1.10935 is checked against the periodic solution from
numerical integration (first lines of `swing.out`); without pumping a1 = 1, so
the rider raises the amplitude by 11 per cent. Already with one harmonic the
truncated system is more accurate than first-order perturbation theory, because
it contains all orders of eps within the kept harmonics. Each further harmonic
gains four to five digits.

`residual.png` shows what is left over: with K = 1 the residual is a pure
cos 3t, with K = 2 a cos 5t, and so on, each smaller by a factor of about 100.

### From the swing to the Earth

| swing | Earth |
|---|---|
| time t | colatitude θ |
| harmonics cos nt | degrees n, Legendre functions P_n^m(cos θ) |
| amplitudes a_n | radial functions U_n(s), V_n(s), W_n(s), P_n(s) (Chebyshev in s) |
| eps cos 2t | flattening, ε P2(cos θ) |
| one equation | momentum equation (3 components) + Poisson's equation |
