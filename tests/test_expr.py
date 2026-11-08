import random
from fractions import Fraction

import pytest

from microreason.verify.expr import (
    MAX_TEXT_LEN,
    Bin,
    ExprError,
    N,
    X,
    evaluate,
    linear_coeffs,
    parse_equation,
    parse_expression,
    render,
    solve_linear,
)


@pytest.mark.parametrize(
    "text,value",
    [
        ("3 + (-5)", -2),
        ("-3 + 5", 2),
        ("2 - 3 * 4", -10),
        ("(2 - 3) * 4", -4),
        ("12 / 5", Fraction(12, 5)),
        ("5 / 7/3", Fraction(5, 21)),  # left-associative division
        ("-14/3 + 1", Fraction(-11, 3)),
        ("8 - 2 - 1", 5),
        ("--4", 4),
    ],
)
def test_expression_values(text, value):
    assert evaluate(parse_expression(text)) == Fraction(value)


@pytest.mark.parametrize(
    "text,solution",
    [
        ("4x + 7 = 31", 6),
        ("3(x + 4) = 27", 5),
        ("-x + 7 = 2", 5),
        ("-3(x - 2) = 9", -1),
        ("5x - 78 = -4x - 249", -19),
        ("x + 4 = 27/5", Fraction(7, 5)),
    ],
)
def test_equation_solutions(text, solution):
    assert solve_linear(*parse_equation(text)) == Fraction(solution)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "3 +",
        "(3 + 4",
        "3 + 4)",
        "2 3",
        "3 ^ 2",
        "import os",
        "1e5",
        "0.5 + 1",
        "3 − 2",  # unicode minus
        "３ + 1",  # full-width digit
        "__import__('os')",
        "(" * 40 + "1" + ")" * 40,
        "1" * 31,
        "1 + " * 70 + "1",
    ],
)
def test_rejects_out_of_grammar(text):
    with pytest.raises(ExprError):
        parse_expression(text)


def test_length_limit():
    assert len("1 + " * 70) > MAX_TEXT_LEN


def test_division_by_zero():
    with pytest.raises(ExprError):
        evaluate(parse_expression("3 / (2 - 2)"))


def test_equation_needs_one_equals():
    for text in ("x + 1", "x = 1 = 2"):
        with pytest.raises(ExprError):
            parse_equation(text)


def test_nonlinear_rejected():
    with pytest.raises(ExprError):
        linear_coeffs(parse_expression("x * x"))


def _random_tree(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return N(rng.randint(-50, 50))
    op = rng.choice("+-*/")
    right = _random_tree(rng, depth - 1)
    if op == "/" and isinstance(right, type(N(0))) and right.value == 0:
        right = N(7)
    return Bin(op, _random_tree(rng, depth - 1), right)


def test_render_parse_roundtrip_property():
    rng = random.Random(0)
    checked = 0
    for _ in range(3000):
        tree = _random_tree(rng, 4)
        try:
            want = evaluate(tree)
        except ExprError:
            continue
        assert evaluate(parse_expression(render(tree))) == want, render(tree)
        checked += 1
    assert checked > 2000


def test_implicit_multiplication_rendering():
    assert render(Bin("*", N(4), X)) == "4x"
    assert render(Bin("*", N(-1), X)) == "-x"
    assert render(Bin("*", N(3), Bin("+", Bin("*", N(1), X), N(4)))) == "3(x + 4)"
