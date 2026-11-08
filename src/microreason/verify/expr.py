"""Restricted arithmetic / linear-equation grammar with exact evaluation.

This is the only parser used on problem text and model output. It never calls eval/exec. The
grammar accepts ASCII digits, the variable x, + - * / ( ) and =, with implicit multiplication
for forms like 4x and 3(x + 4). Input length, nesting depth and literal size are bounded.

    expr    := term (('+' | '-') term)*
    term    := unary (('*' | '/') unary | <implicit> primary)*
    unary   := '-' unary | primary
    primary := NUMBER | 'x' | '(' expr ')'
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

MAX_TEXT_LEN = 256
MAX_DEPTH = 32
MAX_DIGITS = 30

PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
_DIGITS = frozenset("0123456789")
_OPS = frozenset("+-*/()=")


class ExprError(ValueError):
    """Raised for any input outside the grammar or any undefined operation."""


@dataclass(frozen=True)
class Num:
    value: Fraction


@dataclass(frozen=True)
class Var:
    name: str = "x"


@dataclass(frozen=True)
class Neg:
    child: Node


@dataclass(frozen=True)
class Bin:
    op: str
    left: Node
    right: Node


Node = Num | Var | Neg | Bin


def N(v: int | Fraction) -> Num:
    return Num(Fraction(v))


X = Var("x")


# ----------------------------------------------------------------------------- formatting


def fmt_value(v: Fraction) -> str:
    """Canonical answer form: integer, or reduced p/q with the sign on the numerator."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _contains_var(n: Node) -> bool:
    if isinstance(n, Var):
        return True
    if isinstance(n, Neg):
        return _contains_var(n.child)
    if isinstance(n, Bin):
        return _contains_var(n.left) or _contains_var(n.right)
    return False


def _is_implicit(n: Node) -> bool:
    return (
        isinstance(n, Bin)
        and n.op == "*"
        and isinstance(n.left, Num)
        and n.left.value.denominator == 1
        and _contains_var(n.right)
    )


def _prec(n: Node) -> int:
    return PREC[n.op] if isinstance(n, Bin) else 3


def render(node: Node, first: bool = True, parent: str | None = None) -> str:
    """Render a tree so that parsing the result yields the same value.

    `first` is true when the node starts the whole expression (a leading minus needs no parens).
    """
    if isinstance(node, Num):
        v = node.value
        s = fmt_value(v)
        if (v < 0 and not first) or (v.denominator != 1 and parent in ("*", "/")):
            return f"({s})"
        return s
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Neg):
        s = "-" + render(node.child, first=False, parent="neg")
        return s if first else f"({s})"

    if _is_implicit(node):
        c = node.left.value
        coef = "" if c == 1 else "-" if c == -1 else str(c.numerator)
        r = node.right
        rs = r.name if isinstance(r, Var) else f"({render(r)})"
        s = coef + rs
        return s if (first or c >= 0) else f"({s})"

    op = node.op
    if _prec(node.left) < PREC[op]:
        ls = f"({render(node.left)})"
    else:
        ls = render(node.left, first=first, parent=op)
    if isinstance(node.right, Bin) and _prec(node.right) <= PREC[op]:
        rs = f"({render(node.right)})"
    else:
        rs = render(node.right, first=False, parent=op)
    return f"{ls} {op} {rs}"


# ----------------------------------------------------------------------------- parsing


def tokenize(text: str) -> list[tuple[str, object]]:
    if not isinstance(text, str):
        raise ExprError("not_text")
    if len(text) > MAX_TEXT_LEN:
        raise ExprError("too_long")
    toks: list[tuple[str, object]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == " ":
            i += 1
        elif c in _DIGITS:
            j = i
            while j < n and text[j] in _DIGITS:
                j += 1
            if j - i > MAX_DIGITS:
                raise ExprError("literal_too_long")
            toks.append(("num", int(text[i:j])))
            i = j
        elif c == "x":
            toks.append(("var", "x"))
            i += 1
        elif c in _OPS:
            toks.append(("op", c))
            i += 1
        else:
            raise ExprError("bad_char")
    return toks


class _Parser:
    def __init__(self, toks: list[tuple[str, object]]):
        self.toks = toks
        self.i = 0
        self.depth = 0

    def _peek(self) -> tuple[str, object] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _is_op(self, *ops: str) -> bool:
        t = self._peek()
        return t is not None and t[0] == "op" and t[1] in ops

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprError("too_deep")

    def parse(self) -> Node:
        if not self.toks:
            raise ExprError("empty")
        node = self.expr()
        if self.i != len(self.toks):
            raise ExprError("trailing_tokens")
        return node

    def expr(self) -> Node:
        node = self.term()
        while self._is_op("+", "-"):
            op = self.toks[self.i][1]
            self.i += 1
            node = Bin(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while True:
            if self._is_op("*", "/"):
                op = self.toks[self.i][1]
                self.i += 1
                node = Bin(op, node, self.unary())
            elif (t := self._peek()) is not None and (t[0] == "var" or t == ("op", "(")):
                node = Bin("*", node, self.primary())
            else:
                return node

    def unary(self) -> Node:
        if self._is_op("-"):
            self.i += 1
            self._enter()
            node = Neg(self.unary())
            self.depth -= 1
            return node
        return self.primary()

    def primary(self) -> Node:
        t = self._peek()
        if t is None:
            raise ExprError("unexpected_end")
        if t[0] == "num":
            self.i += 1
            return Num(Fraction(t[1]))
        if t[0] == "var":
            self.i += 1
            return X
        if t == ("op", "("):
            self.i += 1
            self._enter()
            node = self.expr()
            if not self._is_op(")"):
                raise ExprError("unbalanced_parens")
            self.i += 1
            self.depth -= 1
            return node
        raise ExprError("unexpected_token")


def parse_expression(text: str) -> Node:
    toks = tokenize(text)
    if ("op", "=") in toks:
        raise ExprError("unexpected_equals")
    return _Parser(toks).parse()


def parse_equation(text: str) -> tuple[Node, Node]:
    toks = tokenize(text)
    eq = [i for i, t in enumerate(toks) if t == ("op", "=")]
    if len(eq) != 1:
        raise ExprError("need_exactly_one_equals")
    k = eq[0]
    return _Parser(toks[:k]).parse(), _Parser(toks[k + 1 :]).parse()


# ----------------------------------------------------------------------------- evaluation


def evaluate(node: Node, x: Fraction | None = None) -> Fraction:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        if x is None:
            raise ExprError("unbound_variable")
        return Fraction(x)
    if isinstance(node, Neg):
        return -evaluate(node.child, x)
    a, b = evaluate(node.left, x), evaluate(node.right, x)
    if node.op == "+":
        return a + b
    if node.op == "-":
        return a - b
    if node.op == "*":
        return a * b
    if b == 0:
        raise ExprError("division_by_zero")
    return a / b


def linear_coeffs(node: Node) -> tuple[Fraction, Fraction]:
    """Return (a, b) with node == a*x + b, raising if node is not linear in x."""
    b = evaluate(node, Fraction(0))
    a = evaluate(node, Fraction(1)) - b
    for probe in (Fraction(2), Fraction(-3), Fraction(7, 2)):
        if evaluate(node, probe) != a * probe + b:
            raise ExprError("not_linear")
    return a, b


def solve_linear(lhs: Node, rhs: Node) -> Fraction:
    a1, b1 = linear_coeffs(lhs)
    a2, b2 = linear_coeffs(rhs)
    a, b = a1 - a2, b1 - b2
    if a == 0:
        raise ExprError("no_unique_solution")
    return -b / a
