# -*- coding: utf-8 -*-
#
# First-order jets: a value carried together with its r- and theta-
# derivatives, so that the divergence of the stress can be evaluated without
# writing out the derivatives by hand (first-order automatic differentiation;
# manual, 'The physics at the nodes').
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""First-order jets: a value carried together with its r- and theta-derivatives, so that the
divergence of the stress can be evaluated without writing out the derivatives by hand
(first-order automatic differentiation; manual, 'The physics at the nodes').

J(v, r, t)  value v with d/dr = r and d/dtheta (at fixed r) = t; +, -, *, / apply the product
            and quotient rules
_D          a dict whose items are also attributes (d.v, d.r, ...)
ZERO        the zero field and its derivatives
"""


class J:
    """First-order jet: value with d/dr and d/dtheta (theta at fixed r)."""
    __slots__ = ('v', 'r', 't')

    def __init__(self, v, r=0.0, t=0.0):
        self.v, self.r, self.t = v, r, t

    @staticmethod
    def _j(o):
        return o if isinstance(o, J) else J(o)

    def __add__(self, o):
        o = self._j(o)
        return J(self.v + o.v, self.r + o.r, self.t + o.t)
    __radd__ = __add__

    def __sub__(self, o):
        o = self._j(o)
        return J(self.v - o.v, self.r - o.r, self.t - o.t)

    def __rsub__(self, o):
        return self._j(o) - self

    def __neg__(self):
        return J(-self.v, -self.r, -self.t)

    def __mul__(self, o):
        o = self._j(o)
        return J(self.v * o.v, self.r * o.v + self.v * o.r, self.t * o.v + self.v * o.t)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._j(o)
        q = self.v / o.v
        return J(q, (self.r - q * o.r) / o.v, (self.t - q * o.t) / o.v)


class _D(dict):
    __getattr__ = dict.__getitem__


ZERO = _D(v=0.0, r=0.0, t=0.0, rr=0.0, rt=0.0, tt=0.0)
