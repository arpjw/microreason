"""Exact-arithmetic problem generators (spec §4).

Each template samples named integer parameters from declared ranges, builds an exact expression
tree, and produces a prompt, worked steps and a Fraction answer. Parameter roles have disjoint
in-distribution and extrapolation magnitude ranges so the extrapolation suite is checkable.

Difficulty levels:
  1  signed addition, subtraction, multiplication
  2  exact rational division, order of operations, one-step equations
  3  multi-step arithmetic, ax + b = c
  4  parentheses, a(x + b) = c, variables on both sides
  5  held-out compositions (never in training)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from microreason.verify.expr import Bin, N, Node, X, fmt_value, render

GENERATOR_VERSION = "1.0"

NumberRange = Literal["in_distribution", "extrapolation"]
Kind = Literal["expression", "linear_equation"]

# Magnitude ranges per parameter role. In-distribution and extrapolation never overlap.
RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "operand": {"in_distribution": (0, 99), "extrapolation": (100, 999)},
    "coef": {"in_distribution": (2, 12), "extrapolation": (13, 40)},
    "solution": {"in_distribution": (0, 20), "extrapolation": (21, 200)},
}

# Probability that an equation is built from an integer solution (otherwise the right-hand side
# is sampled and the solution may be a fraction).
P_INTEGER_SOLUTION = 0.7

EXPR_PROMPTS = ("Compute: {m}", "Evaluate: {m}", "What is {m}?", "Calculate: {m}")
EQ_PROMPTS = (
    "Solve for x: {m}",
    "Find x: {m}",
    "Solve the equation {m} for x.",
    "What value of x satisfies {m}?",
)


class GenerationReject(Exception):
    """The sampled parameters do not give a valid problem; the caller draws a new seed."""


class Sampler:
    def __init__(self, rng: random.Random, number_range: NumberRange):
        self.rng = rng
        self.range = number_range
        self.params: dict[str, int] = {}

    def draw(self, name: str, role: str, nonzero: bool = False) -> int:
        lo, hi = RANGES[role][self.range]
        if nonzero and lo == 0:
            lo = 1
        mag = self.rng.randint(lo, hi)
        v = -mag if self.rng.random() < 0.5 else mag
        self.params[name] = v
        return v

    def coin(self, p: float) -> bool:
        return self.rng.random() < p


@dataclass
class Problem:
    kind: Kind
    math_text: str
    prompt: str
    steps: list[str]
    answer: Fraction
    tree: Node | None = None
    lhs: Node | None = None
    rhs: Node | None = None
    params: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Template:
    template_id: str
    family: str
    difficulty: int
    roles: dict[str, str]
    fn: Callable[[Sampler], Problem]
    heldout: bool = False


# ----------------------------------------------------------------------------- helpers

_VERBS = {"+": "Add", "-": "Subtract", "*": "Multiply", "/": "Divide"}


def _apply(op: str, a: Fraction, b: Fraction) -> Fraction:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise GenerationReject("division_by_zero")
    return a / b


def _eval_with_steps(node: Node, steps: list[str]) -> Fraction:
    if not isinstance(node, Bin):
        return node.value  # Num
    a = _eval_with_steps(node.left, steps)
    b = _eval_with_steps(node.right, steps)
    v = _apply(node.op, a, b)
    steps.append(f"{_VERBS[node.op]}: {render(Bin(node.op, N(a), N(b)))} = {fmt_value(v)}")
    return v


def _expr_problem(tree: Node, s: Sampler) -> Problem:
    steps: list[str] = []
    value = _eval_with_steps(tree, steps)
    text = render(tree)
    prompt = s.rng.choice(EXPR_PROMPTS).format(m=text)
    return Problem("expression", text, prompt, steps, value, tree=tree, params=dict(s.params))


def lin_side(a: int | Fraction, b: int | Fraction) -> Node:
    """Build a*x + b, writing a negative constant as subtraction."""
    a, b = Fraction(a), Fraction(b)
    if a == 0:
        return N(b)
    term = Bin("*", N(a), X)
    if b > 0:
        return Bin("+", term, N(b))
    if b < 0:
        return Bin("-", term, N(-b))
    return term


def plus_const(node: Node, c: int | Fraction) -> Node:
    c = Fraction(c)
    if c > 0:
        return Bin("+", node, N(c))
    if c < 0:
        return Bin("-", node, N(-c))
    return node


def _x_term(a: int | Fraction) -> str:
    return render(lin_side(a, 0))


def finish_linear(a: Fraction, b: Fraction, c: Fraction) -> tuple[list[str], Fraction]:
    """Steps that solve a*x + b = c, assuming a != 0."""
    a, b, c = Fraction(a), Fraction(b), Fraction(c)
    steps: list[str] = []
    rhs = c
    if b != 0:
        verb = f"Subtract {fmt_value(b)} from" if b > 0 else f"Add {fmt_value(-b)} to"
        rhs = c - b
        steps.append(f"{verb} both sides: {_x_term(a)} = {fmt_value(rhs)}")
    x = rhs / a
    if a != 1:
        steps.append(f"Divide both sides by {fmt_value(a)}: x = {fmt_value(x)}")
    return steps, x


def _eq_problem(lhs: Node, rhs: Node, steps: list[str], answer: Fraction, s: Sampler) -> Problem:
    text = f"{render(lhs)} = {render(rhs)}"
    prompt = s.rng.choice(EQ_PROMPTS).format(m=text)
    if not steps or steps[-1].split(": ", 1)[1] != f"x = {fmt_value(answer)}":
        raise GenerationReject("trace_does_not_end_in_answer")
    return Problem(
        "linear_equation", text, prompt, steps, answer, lhs=lhs, rhs=rhs, params=dict(s.params)
    )


def _move_x_term(a: Fraction, b: Fraction, c: Fraction, d: Fraction) -> str:
    """Step text for a*x + b = c*x + d  ->  (a - c)*x + b = d."""
    verb = f"Subtract {_x_term(c)} from" if c > 0 else f"Add {_x_term(-c)} to"
    return f"{verb} both sides: {render(lin_side(a - c, b))} = {fmt_value(d)}"


# ----------------------------------------------------------------------------- level 1


def t_add(s: Sampler) -> Problem:
    return _expr_problem(Bin("+", N(s.draw("a", "operand")), N(s.draw("b", "operand"))), s)


def t_sub(s: Sampler) -> Problem:
    return _expr_problem(Bin("-", N(s.draw("a", "operand")), N(s.draw("b", "operand"))), s)


def t_mul(s: Sampler) -> Problem:
    return _expr_problem(Bin("*", N(s.draw("a", "operand")), N(s.draw("b", "operand"))), s)


# ----------------------------------------------------------------------------- level 2


def t_div(s: Sampler) -> Problem:
    a = s.draw("a", "operand")
    b = s.draw("b", "operand", nonzero=True)
    return _expr_problem(Bin("/", N(a), N(b)), s)


def t_add_mul(s: Sampler) -> Problem:
    a, b, c = (s.draw(k, "operand") for k in "abc")
    return _expr_problem(Bin("+", N(a), Bin("*", N(b), N(c))), s)


def t_one_step_add(s: Sampler) -> Problem:
    a = s.draw("a", "operand", nonzero=True)
    x = Fraction(s.draw("x", "operand"))
    b = x + a
    steps, sol = finish_linear(Fraction(1), Fraction(a), b)
    return _eq_problem(lin_side(1, a), N(b), steps, sol, s)


def t_one_step_mul(s: Sampler) -> Problem:
    a = s.draw("a", "coef")
    if s.coin(P_INTEGER_SOLUTION):
        b = a * Fraction(s.draw("x", "operand"))
    else:
        b = Fraction(s.draw("b", "operand", nonzero=True))
    steps, sol = finish_linear(Fraction(a), Fraction(0), b)
    return _eq_problem(lin_side(a, 0), N(b), steps, sol, s)


# ----------------------------------------------------------------------------- level 3


def t_mul_sub_mul(s: Sampler) -> Problem:
    a, b, c, d = (s.draw(k, "operand") for k in "abcd")
    return _expr_problem(Bin("-", Bin("*", N(a), N(b)), Bin("*", N(c), N(d))), s)


def t_sub_mul_add(s: Sampler) -> Problem:
    a, b, c, d = (s.draw(k, "operand") for k in "abcd")
    return _expr_problem(Bin("+", Bin("-", N(a), Bin("*", N(b), N(c))), N(d)), s)


def t_linear_ax_b_c(s: Sampler) -> Problem:
    a = s.draw("a", "coef")
    b = s.draw("b", "operand", nonzero=True)
    if s.coin(P_INTEGER_SOLUTION):
        c = a * Fraction(s.draw("x", "solution")) + b
    else:
        c = Fraction(s.draw("c", "operand"))
    steps, sol = finish_linear(Fraction(a), Fraction(b), c)
    return _eq_problem(lin_side(a, b), N(c), steps, sol, s)


# ----------------------------------------------------------------------------- level 4


def t_paren_mul(s: Sampler) -> Problem:
    a, b, c, d = (s.draw(k, "operand") for k in "abcd")
    return _expr_problem(Bin("*", Bin("+", N(a), N(b)), Bin("-", N(c), N(d))), s)


def t_mul_paren_sub(s: Sampler) -> Problem:
    a, b, c = (s.draw(k, "operand") for k in "abc")
    return _expr_problem(Bin("*", N(a), Bin("-", N(b), N(c))), s)


def t_linear_paren(s: Sampler) -> Problem:
    a = s.draw("a", "coef")
    b = s.draw("b", "operand", nonzero=True)
    if s.coin(P_INTEGER_SOLUTION):
        c = a * (Fraction(s.draw("x", "solution")) + b)
    else:
        c = Fraction(s.draw("c", "operand"))
    inner = lin_side(1, b)
    q = c / a
    steps = [f"Divide both sides by {a}: {render(inner)} = {fmt_value(q)}"]
    more, sol = finish_linear(Fraction(1), Fraction(b), q)
    return _eq_problem(Bin("*", N(a), inner), N(c), steps + more, sol, s)


def t_linear_both_sides(s: Sampler) -> Problem:
    a = s.draw("a", "coef")
    c = s.draw("c", "coef")
    if c == a:
        raise GenerationReject("equal_coefficients")
    b = s.draw("b", "operand", nonzero=True)
    if s.coin(P_INTEGER_SOLUTION):
        d = (a - c) * Fraction(s.draw("x", "solution")) + b
    else:
        d = Fraction(s.draw("d", "operand"))
    steps = [_move_x_term(Fraction(a), Fraction(b), Fraction(c), d)]
    more, sol = finish_linear(Fraction(a - c), Fraction(b), d)
    return _eq_problem(lin_side(a, b), lin_side(c, d), steps + more, sol, s)


# ----------------------------------------------------------------------------- level 5 (held out)


def h_paren_div_add(s: Sampler) -> Problem:
    a, b, c, d, e = (s.draw(k, "operand") for k in "abcde")
    if c == d:
        raise GenerationReject("zero_denominator")
    tree = Bin("+", Bin("/", Bin("+", N(a), N(b)), Bin("-", N(c), N(d))), N(e))
    return _expr_problem(tree, s)


def h_nested(s: Sampler) -> Problem:
    a, b, c, d, e = (s.draw(k, "operand") for k in "abcde")
    tree = Bin("-", Bin("*", N(a), Bin("+", N(b), Bin("*", N(c), N(d)))), N(e))
    return _expr_problem(tree, s)


def h_linear_paren_both(s: Sampler) -> Problem:
    a = s.draw("a", "coef")
    d = s.draw("d", "coef")
    if d == a:
        raise GenerationReject("equal_coefficients")
    b = s.draw("b", "operand", nonzero=True)
    c = s.draw("c", "operand", nonzero=True)
    if s.coin(P_INTEGER_SOLUTION):
        x = Fraction(s.draw("x", "solution"))
        e = a * (x + b) + c - d * x
    else:
        e = Fraction(s.draw("e", "operand"))
    lhs = plus_const(Bin("*", N(a), lin_side(1, b)), c)
    rhs = lin_side(d, e)
    k = Fraction(a * b + c)
    steps = [
        f"Expand: {render(plus_const(lin_side(a, a * b), c))} = {render(rhs)}",
        f"Combine constants: {render(lin_side(a, k))} = {render(rhs)}",
        _move_x_term(Fraction(a), k, Fraction(d), e),
    ]
    more, sol = finish_linear(Fraction(a - d), k, e)
    return _eq_problem(lhs, rhs, steps + more, sol, s)


# ----------------------------------------------------------------------------- registry

_OP2 = {"a": "operand", "b": "operand"}
_OP3 = {"a": "operand", "b": "operand", "c": "operand"}
_OP4 = {k: "operand" for k in "abcd"}
_OP5 = {k: "operand" for k in "abcde"}

TEMPLATES: tuple[Template, ...] = (
    Template("add_ab", "arithmetic", 1, _OP2, t_add),
    Template("sub_ab", "arithmetic", 1, _OP2, t_sub),
    Template("mul_ab", "arithmetic", 1, _OP2, t_mul),
    Template("div_ab", "division", 2, _OP2, t_div),
    Template("add_mul_abc", "order_of_operations", 2, _OP3, t_add_mul),
    Template("x_plus_a_equals_b", "one_step_equation", 2, {"a": "operand", "x": "operand"},
             t_one_step_add),
    Template("ax_equals_b", "one_step_equation", 2,
             {"a": "coef", "x": "operand", "b": "operand"}, t_one_step_mul),
    Template("mul_sub_mul", "multi_step_arithmetic", 3, _OP4, t_mul_sub_mul),
    Template("sub_mul_add", "multi_step_arithmetic", 3, _OP4, t_sub_mul_add),
    Template("ax_plus_b_equals_c", "linear_equation", 3,
             {"a": "coef", "b": "operand", "x": "solution", "c": "operand"}, t_linear_ax_b_c),
    Template("paren_sum_times_diff", "parentheses", 4, _OP4, t_paren_mul),
    Template("a_times_paren_diff", "parentheses", 4, _OP3, t_mul_paren_sub),
    Template("a_paren_x_plus_b_equals_c", "linear_equation", 4,
             {"a": "coef", "b": "operand", "x": "solution", "c": "operand"}, t_linear_paren),
    Template("ax_plus_b_equals_cx_plus_d", "linear_equation", 4,
             {"a": "coef", "c": "coef", "b": "operand", "x": "solution", "d": "operand"},
             t_linear_both_sides),
    Template("heldout_paren_quotient_plus", "parentheses", 5, _OP5, h_paren_div_add,
             heldout=True),
    Template("heldout_nested_product", "multi_step_arithmetic", 5, _OP5, h_nested,
             heldout=True),
    Template("heldout_paren_both_sides", "linear_equation", 5,
             {"a": "coef", "d": "coef", "b": "operand", "c": "operand", "x": "solution",
              "e": "operand"}, h_linear_paren_both, heldout=True),
)  # fmt: skip

TEMPLATES_BY_ID = {t.template_id: t for t in TEMPLATES}
