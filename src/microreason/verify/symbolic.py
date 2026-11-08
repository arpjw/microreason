"""Independent SymPy check of generated problems (spec §4).

The problem text is re-parsed with the restricted grammar and converted to SymPy objects, so the
check is independent of the generator's own tree and Fraction arithmetic. Inputs are bounded by
the grammar's length and depth limits, so no separate timeout is needed for these small forms.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from microreason.verify.expr import Neg, Node, Num, Var, parse_equation, parse_expression

_X = sympy.Symbol("x")


def to_sympy(node: Node) -> sympy.Expr:
    if isinstance(node, Num):
        return sympy.Rational(node.value.numerator, node.value.denominator)
    if isinstance(node, Var):
        return _X
    if isinstance(node, Neg):
        return -to_sympy(node.child)
    a, b = to_sympy(node.left), to_sympy(node.right)
    if node.op == "+":
        return a + b
    if node.op == "-":
        return a - b
    if node.op == "*":
        return a * b
    if b == 0:
        raise ZeroDivisionError("division by zero in problem text")
    return a / b


def _to_fraction(r: sympy.Expr) -> Fraction:
    if not r.is_Rational:
        raise ValueError(f"not an exact rational: {r}")
    return Fraction(int(r.p), int(r.q))


def sympy_expression_value(text: str) -> Fraction:
    return _to_fraction(to_sympy(parse_expression(text)))


def sympy_linear_solution(text: str) -> Fraction:
    lhs, rhs = parse_equation(text)
    poly = sympy.Poly(sympy.expand(to_sympy(lhs) - to_sympy(rhs)), _X)
    if poly.degree() != 1:
        raise ValueError("equation is not linear with a unique solution")
    c1, c0 = poly.coeff_monomial(_X), poly.coeff_monomial(1)
    return _to_fraction(-c0 / c1)
