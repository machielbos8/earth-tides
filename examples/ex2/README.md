## Example 2: The loaded elliptical drum

The method of the paper in two dimensions (Sect. 3.5): level-curve coordinates, Chebyshev
collocation in the radial coordinate and Galerkin projection in angle, checked against an exact
solution. No Earth model is needed; the script uses only NumPy.

### The problem

A drum skin is stretched over an elliptical frame and loaded. Its deflection u satisfies

    -lap u = load   inside x^2/a^2 + y^2/b^2 <= 1,   u = 0 on the rim,

with a = 1 and b = 1 - f (f is the flattening). Two loads:

* a uniform pressure, load = 1;
* a quadrupole, load = x^2 - y^2, the analogue of the degree-2 tide.

Both have exact polynomial solutions. With A = 1/a^2 + 1/b^2 the uniform-load solution is
u = (1 - x^2/a^2 - y^2/b^2) / (2A). The quadrupole solution is (1 - x^2/a^2 - y^2/b^2)(alpha +
beta x^2 + gamma y^2), with alpha, beta, gamma from a 3 x 3 linear system (`exact` in drum.py).

### The procedure

1. **Level-curve coordinates.** Write r = s R(theta), where R(theta) is the radius of the frame.
   The rim becomes s = 1, and every curve s = const is a similar ellipse, as the level surfaces of
   the Earth are. The Laplacian becomes (q = R'/R)

       R^2 lap u = (1 + q^2) u_ss + (1 + q^2 - q') u_s / s + u_tt / s^2 - 2 q u_st / s .

   The flattening appears only through q ~ f sin 2 theta. This term couples cos 2k theta with
   cos 2(k +- 1) theta, as mu_2 P2 couples degrees on the Earth.
2. **Trial function.** u = sum_k U_k(s) cos 2k theta, k = 0 .. K-1.
3. **Galerkin projection in theta.** The residual may contain none of the kept harmonics. The
   Fourier coefficients are taken numerically, so no coupling coefficient is derived by hand.
4. **Chebyshev collocation in s.** The projected equations hold at the interior Gauss-Lobatto
   nodes of [0, 1]. The boundary conditions are:
   * at the rim, U_k(1) = 0;
   * at the centre, U_0'(0) = 0 and U_k(0) = 0 for k > 0 (regularity).

### Result (`drum.out`)

* **Uniform load:** exact to round-off with ONE harmonic, at any flattening. The
  coordinates absorb the shape: in them the solution is (1 - s^2)/(2A).
* **Quadrupole load:** the error falls exponentially with K. With f = 0.1:

  | K | error |
  |---|---|
  | 2 | 4e-2 |
  | 4 | 1e-4 |
  | 8 | 9e-10 |

  With the Earth's flattening (1/300), K = 6 reaches round-off.
* **First-order perturbation theory** in f is limited to errors of order f^2: 5e-5 at f = 1/300,
  4e-2 at f = 0.1. The truncated system contains all orders of f within the kept harmonics.

Run: `python drum.py` (2 s).
