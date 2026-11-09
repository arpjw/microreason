"""Generator v3: compositional expression trees and linear equations (reports/protocol_v3.md).

Problems are sampled from a grammar instead of fixed templates.

**Expressions** are random binary trees with 2–6 integer leaves and an operator from `+ - * /`
at every internal node. Topologies come from recursive random splits: the left subtree gets a
uniformly drawn number of leaves. Traces evaluate the tree post-order, innermost first and left
to right, with the same step wording as generator v1: `Add: a + b = c`.

**Linear equations** are `L = R`. The left side contains x; the right side may or may not.
- **Each side** is a term, optionally plus or minus a constant.
- **Terms:** `x`, `kx`, `k(x + b)`, `k(x - b)`.
- **A right side** may also be a bare constant.
- **Trace order:** expand each parenthesized side (`Expand`), combine its constants
  (`Combine constants`), move the right side's x term, move the constant, then divide.
  This is generator v1's step vocabulary.

**Shape.** A problem's shape is its structure with the numbers abstracted away:
- expressions: `(+ n (* n n))`;
- equations: `k(x+b)+c=kx-c`.

Shapes are split into train-visible, held-out-val and held-out-test sets by a fixed hash rule
(`shape_set`), before any rows exist. Only *eligible* shapes are ever held out: expressions with
at least 4 leaves, and equations of structural complexity at least 4. `coverage_report` checks
that every operator pairing and step type in the held-out sets is also train-visible.

**Number ranges** (operand magnitudes, chosen by digit count):

| Range | Digits |
| --- | --- |
| in distribution | 1–3 (0–999) |
| extrapolation | 4–5 (1,000–99,999) |

**Size caps.** Intermediate values are capped, so traces stay readable and bounded in length.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Literal

from microreason.data.generators import (
    EQ_PROMPTS,
    EXPR_PROMPTS,
    GenerationReject,
    Problem,
    _eval_with_steps,
    _move_x_term,
    finish_linear,
    lin_side,
    plus_const,
)
from microreason.verify.expr import Bin, N, Node, Num, X, fmt_value, render

GENERATOR_VERSION_V3 = "3.0"
NumberRange = Literal["in_distribution", "extrapolation"]
ShapeSet = Literal["train", "heldout_val", "heldout_test"]

OPS = ("+", "-", "*", "/")
DIGIT_RANGES: dict[str, dict[int, tuple[int, int]]] = {
    "in_distribution": {1: (0, 9), 2: (10, 99), 3: (100, 999)},
    "extrapolation": {4: (1000, 9999), 5: (10000, 99999)},
}
DIGIT_WEIGHTS: dict[str, dict[int, float]] = {
    "in_distribution": {1: 0.35, 2: 0.40, 3: 0.25},
    "extrapolation": {4: 0.6, 5: 0.4},
}
COEF_RANGES = {"in_distribution": (2, 12), "extrapolation": (13, 40)}
SOLUTION_RANGES = {"in_distribution": (0, 20), "extrapolation": (21, 200)}
LEAF_WEIGHTS = {2: 0.15, 3: 0.25, 4: 0.25, 5: 0.20, 6: 0.15}
MAX_LEAVES = max(LEAF_WEIGHTS)
# Caps on |numerator| and denominator of every intermediate value.
VALUE_CAP = {"in_distribution": 10**6, "extrapolation": 10**12}
DENOM_CAP = {"in_distribution": 10**4, "extrapolation": 10**6}
P_EXPRESSION = 0.6
P_INTEGER_SOLUTION = 0.7
HELDOUT_FRACTIONS = {"heldout_val": 0.10, "heldout_test": 0.10}
MIN_ELIGIBLE_LEAVES = 4
MIN_ELIGIBLE_EQ_COMPLEXITY = 4

TERM_FORMS = ("x", "kx", "k(x+b)", "k(x-b)")
CONST_SUFFIXES = ("", "+c", "-c")


# ----------------------------------------------------------------------------- shapes


def expr_shape(node: Node) -> str:
    if isinstance(node, Num):
        return "n"
    if not isinstance(node, Bin):
        raise ValueError("expression shapes only cover numeric trees")
    return f"({node.op} {expr_shape(node.left)} {expr_shape(node.right)})"


def shape_leaves(shape: str) -> int:
    return shape.count("n") if shape.startswith("(") or shape == "n" else 0


def eq_complexity(shape: str) -> int:
    """Structural units of an equation shape: parentheses, constants, and an x on the right."""
    left, right = shape.split("=")
    units = left.count("(") + right.count("(")
    units += left.count("c") + right.count("c")
    units += 1 if "x" in right else 0
    units += 1 if left.startswith("k") else 0  # a coefficient to divide by
    return units


def is_eligible(shape: str) -> bool:
    if shape.startswith("("):
        return shape_leaves(shape) >= MIN_ELIGIBLE_LEAVES
    return eq_complexity(shape) >= MIN_ELIGIBLE_EQ_COMPLEXITY


def shape_set(shape: str) -> ShapeSet:
    """Fixed, seedless assignment of a shape to train-visible or one of two held-out sets."""
    if not is_eligible(shape):
        return "train"
    u = int(hashlib.sha256(f"microreason-v3-shape:{shape}".encode()).hexdigest()[:12], 16)
    u /= 16**12
    if u < HELDOUT_FRACTIONS["heldout_val"]:
        return "heldout_val"
    if u < HELDOUT_FRACTIONS["heldout_val"] + HELDOUT_FRACTIONS["heldout_test"]:
        return "heldout_test"
    return "train"


def all_expr_shapes(max_leaves: int = MAX_LEAVES) -> list[str]:
    def topo(n: int) -> list[str]:
        if n == 1:
            return ["n"]
        out = []
        for k in range(1, n):
            for left, right in product(topo(k), topo(n - k)):
                for op in OPS:
                    out.append(f"({op} {left} {right})")
        return out

    return [s for n in range(2, max_leaves + 1) for s in topo(n)]


def all_eq_shapes() -> list[str]:
    lefts = [t + c for t in TERM_FORMS for c in CONST_SUFFIXES]
    rights = [*lefts, "c"]
    return [f"{lft}={rgt}" for lft in lefts for rgt in rights]


def expr_features(shape: str) -> set[str]:
    """Operators and (parent op, side, child op) pairs, read from a shape string."""
    feats: set[str] = set()

    def parse(i: int) -> tuple[str, int]:  # returns (op or "n", next index)
        if shape[i] == "n":
            return "n", i + 1
        op = shape[i + 1]
        lo, j = parse(i + 3)
        ro, j = parse(j + 1)
        feats.add(f"op:{op}")
        feats.add(f"pair:{op}<{lo}")
        feats.add(f"pair:{op}>{ro}")
        return op, j + 1

    parse(0)
    return feats


def eq_features(shape: str) -> set[str]:
    left, right = shape.split("=")
    return {f"left:{left}", f"right:{right}"} | eq_step_types(shape)


def eq_step_types(shape: str) -> set[str]:
    left, right = shape.split("=")
    steps = {"Divide"} if left.startswith("k") or "x" in right else set()
    if "(" in left or "(" in right:
        steps.add("Expand")
    if ("(" in left and "c" in left) or ("(" in right and "c" in right):
        steps.add("Combine")
    if "x" in right:
        steps.add("MoveX")
    if "c" in left or "(" in left:
        steps.add("MoveConst")
    return steps


def coverage_report() -> dict:
    """Check that held-out shapes recombine train-visible parts only."""
    out: dict = {"passed": True, "failures": []}
    for kind, shapes, feats in (
        ("expression", all_expr_shapes(), expr_features),
        ("equation", all_eq_shapes(), eq_features),
    ):
        sets: dict[str, list[str]] = {"train": [], "heldout_val": [], "heldout_test": []}
        for s in shapes:
            sets[shape_set(s)].append(s)
        train_feats = set().union(*(feats(s) for s in sets["train"]))
        info = {k: len(v) for k, v in sets.items()}
        info["eligible"] = sum(is_eligible(s) for s in shapes)
        for hs in ("heldout_val", "heldout_test"):
            held = set().union(*(feats(s) for s in sets[hs])) if sets[hs] else set()
            missing = held - train_feats
            info[f"{hs}_features_not_in_train"] = sorted(missing)
            if missing:
                out["passed"] = False
                out["failures"].append(f"{kind} {hs}: {sorted(missing)}")
            if not sets[hs]:
                out["passed"] = False
                out["failures"].append(f"{kind} {hs}: empty")
        out[kind] = info
    return out


# ----------------------------------------------------------------------------- sampling


@dataclass
class TreeSampler:
    rng: random.Random
    number_range: NumberRange
    params: dict[str, int] = field(default_factory=dict)

    def _name(self) -> str:
        i = len(self.params)
        return chr(ord("a") + i) if i < 26 else f"p{i}"

    def operand(self, nonzero: bool = False, signed: bool = True) -> int:
        w = DIGIT_WEIGHTS[self.number_range]
        digits = self.rng.choices(list(w), weights=list(w.values()))[0]
        lo, hi = DIGIT_RANGES[self.number_range][digits]
        if nonzero and lo == 0:
            lo = 1
        v = self.rng.randint(lo, hi)
        if signed and self.rng.random() < 0.5:
            v = -v
        self.params[self._name()] = v
        return v

    def coef(self) -> int:
        lo, hi = COEF_RANGES[self.number_range]
        v = self.rng.randint(lo, hi) * (-1 if self.rng.random() < 0.3 else 1)
        self.params[self._name()] = v
        return v

    def solution(self) -> int:
        lo, hi = SOLUTION_RANGES[self.number_range]
        v = self.rng.randint(lo, hi) * (-1 if self.rng.random() < 0.5 else 1)
        self.params["x"] = v
        return v


def _check_caps(v: Fraction, nr: NumberRange) -> None:
    if abs(v.numerator) > VALUE_CAP[nr] or v.denominator > DENOM_CAP[nr]:
        raise GenerationReject("value_cap")


def _random_tree(s: TreeSampler, leaves: int) -> Node:
    if leaves == 1:
        return N(s.operand())
    k = s.rng.randint(1, leaves - 1)
    op = s.rng.choice(OPS)
    return Bin(op, _random_tree(s, k), _random_tree(s, leaves - k))


def _capped_eval(node: Node, nr: NumberRange) -> Fraction:
    if isinstance(node, Num):
        return node.value
    a, b = _capped_eval(node.left, nr), _capped_eval(node.right, nr)
    if node.op == "/" and b == 0:
        raise GenerationReject("division_by_zero")
    v = {"+": a + b, "-": a - b, "*": a * b}.get(node.op) if node.op != "/" else a / b
    _check_caps(v, nr)
    return v


def sample_expression(s: TreeSampler) -> Problem:
    w = LEAF_WEIGHTS
    leaves = s.rng.choices(list(w), weights=list(w.values()))[0]
    tree = _random_tree(s, leaves)
    _capped_eval(tree, s.number_range)
    steps: list[str] = []
    value = _eval_with_steps(tree, steps)
    text = render(tree)
    prompt = s.rng.choice(EXPR_PROMPTS).format(m=text)
    return Problem("expression", text, prompt, steps, value, tree=tree, params=dict(s.params))


def _term(form: str, k: int, b: int) -> Node:
    if form == "x":
        return X
    if form == "kx":
        return Bin("*", N(k), X)
    inner = Bin("+" if form == "k(x+b)" else "-", X, N(b))
    return Bin("*", N(k), inner)


def _side(form: str, k: int, b: int, c: int) -> Node:
    """form is a term form plus a constant suffix, or "c" for a bare constant."""
    if form == "c":
        return N(c)
    term_form = form[:-2] if form.endswith(("+c", "-c")) else form
    node = _term(term_form, k, b)
    if form.endswith("+c"):
        return Bin("+", node, N(c))
    if form.endswith("-c"):
        return Bin("-", node, N(c))
    return node


def _side_coeffs(form: str, k: int, b: int, c: int) -> tuple[Fraction, Fraction]:
    """(x coefficient, constant) of a side."""
    if form == "c":
        return Fraction(0), Fraction(c)
    term_form = form[:-2] if form.endswith(("+c", "-c")) else form
    a = {"x": 1, "kx": k, "k(x+b)": k, "k(x-b)": k}[term_form]
    const = {"x": 0, "kx": 0, "k(x+b)": k * b, "k(x-b)": -k * b}[term_form]
    const += c if form.endswith("+c") else (-c if form.endswith("-c") else 0)
    return Fraction(a), Fraction(const)


def _expanded(form: str, k: int, b: int, c: int) -> Node:
    """A parenthesized side after `Expand`, before constants are combined."""
    term_form = form[:-2] if form.endswith(("+c", "-c")) else form
    kb = k * b if term_form == "k(x+b)" else -k * b
    node = lin_side(k, kb)
    if form.endswith("+c"):
        return plus_const(node, c)
    if form.endswith("-c"):
        return plus_const(node, -c)
    return node


def _eq_trace(lf, lp, rf, rp) -> tuple[list[str], Fraction]:
    """Canonical solution steps for L = R, with (k, b, c) parameters for each side."""
    steps: list[str] = []
    lhs_node, rhs_node = _side(lf, *lp), _side(rf, *rp)
    if "(" in lf:
        lhs_node = _expanded(lf, *lp)
        steps.append(f"Expand: {render(lhs_node)} = {render(rhs_node)}")
    if "(" in rf:
        rhs_node = _expanded(rf, *rp)
        steps.append(f"Expand: {render(lhs_node)} = {render(rhs_node)}")
    la, lc = _side_coeffs(lf, *lp)
    ra, rc = _side_coeffs(rf, *rp)
    if "(" in lf and lf.endswith(("+c", "-c")):
        lhs_node = lin_side(la, lc)
        steps.append(f"Combine constants: {render(lhs_node)} = {render(rhs_node)}")
    if "(" in rf and rf.endswith(("+c", "-c")):
        rhs_node = lin_side(ra, rc)
        steps.append(f"Combine constants: {render(lhs_node)} = {render(rhs_node)}")
    if ra != 0:
        if la == ra:
            raise GenerationReject("no_unique_solution")
        steps.append(_move_x_term(la, lc, ra, rc))
        la, ra = la - ra, Fraction(0)
    more, x = finish_linear(la, lc, rc)
    return steps + more, x


def _term_part(form: str) -> str:
    return form[:-2] if form.endswith(("+c", "-c")) else form


def sample_equation(s: TreeSampler) -> Problem:
    """Sample forms, then parameters. With probability P_INTEGER_SOLUTION an integer x is drawn
    first and the last constant (the right side's if it has one, else the left's) is solved
    for; its sign decides between the `+c` and `-c` forms, so the realised shape is the one
    whose constant has that sign."""
    lf = s.rng.choice(TERM_FORMS) + s.rng.choice(CONST_SUFFIXES)
    rf = s.rng.choice([*(t + c for t in TERM_FORMS for c in CONST_SUFFIXES), "c"])

    def side_params(form: str, solved: bool) -> list[int]:
        k = s.coef() if "k" in form else 1
        b = s.operand(nonzero=True, signed=False) if "(" in form else 0
        if form == "c":
            c = 0 if solved else s.operand()
        elif "c" in form:
            c = 0 if solved else s.operand(nonzero=True, signed=False)
        else:
            c = 0
        return [k, b, c]

    integer = s.rng.random() < P_INTEGER_SOLUTION
    solve_right = "c" in rf
    solve_left = integer and not solve_right and "c" in lf
    lp = side_params(lf, solved=solve_left)
    rp = side_params(rf, solved=integer and solve_right)
    if integer and (solve_right or solve_left):
        x = Fraction(s.solution())
        if solve_right:
            la, lc = _side_coeffs(lf, *lp)
            ra, rc = _side_coeffs(_term_part(rf) if rf != "c" else "c", *rp)
            signed = (la * x + lc) - (ra * x + rc)
        else:
            ra, rc = _side_coeffs(rf, *rp)
            la, lc = _side_coeffs(_term_part(lf), *lp)
            signed = (ra * x + rc) - (la * x + lc)
        hi = DIGIT_RANGES[s.number_range][max(DIGIT_RANGES[s.number_range])][1]
        lo = 1 if s.number_range == "in_distribution" else (
            DIGIT_RANGES["extrapolation"][min(DIGIT_RANGES["extrapolation"])][0])  # fmt: skip
        mag = abs(signed)
        if signed.denominator != 1 or (signed == 0 and (solve_left or rf != "c")):
            raise GenerationReject("solved_constant_out_of_range")
        if mag > hi or (mag < lo and not (rf == "c" and solve_right and mag == 0)):
            raise GenerationReject("solved_constant_out_of_range")
        if solve_right and rf == "c":
            rp[2] = int(signed)
        elif solve_right:
            rf = _term_part(rf) + ("+c" if signed > 0 else "-c")
            rp[2] = int(mag)
        else:
            lf = _term_part(lf) + ("+c" if signed > 0 else "-c")
            lp[2] = int(mag)
        s.params[s._name()] = int(signed)
    lhs, rhs = _side(lf, *lp), _side(rf, *rp)
    steps, x_val = _eq_trace(lf, lp, rf, rp)
    _check_caps(x_val, s.number_range)
    text = f"{render(lhs)} = {render(rhs)}"
    prompt = s.rng.choice(EQ_PROMPTS).format(m=text)
    if not steps or steps[-1].split(": ", 1)[1] != f"x = {fmt_value(x_val)}":
        raise GenerationReject("trace_does_not_end_in_answer")
    p = Problem("linear_equation", text, prompt, steps, x_val, lhs=lhs, rhs=rhs,
                params=dict(s.params))  # fmt: skip
    p.eq_shape = f"{lf}={rf}"  # type: ignore[attr-defined]
    return p


def sample_problem(rng: random.Random, number_range: NumberRange) -> tuple[Problem, str]:
    """One problem and its shape. Raises GenerationReject for invalid draws."""
    s = TreeSampler(rng, number_range)
    if rng.random() < P_EXPRESSION:
        p = sample_expression(s)
        return p, expr_shape(p.tree)
    p = sample_equation(s)
    return p, p.eq_shape  # type: ignore[attr-defined]


def difficulty(shape: str) -> int:
    """Level 1–5 from structure (spec §4): ops for expressions, complexity for equations."""
    if shape.startswith("("):
        ops = shape_leaves(shape) - 1
        if ops == 1:
            return 2 if shape.startswith("(/") else 1
        return min(5, ops)
    return min(5, max(2, eq_complexity(shape)))


def family(shape: str) -> str:
    if shape.startswith("("):
        return f"expression_{shape_leaves(shape) - 1}op"
    return "linear_equation"
