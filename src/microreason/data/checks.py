"""Record-level checks used at build time and again by the audit.

independent_check: the answer recomputed by SymPy from the problem text must equal the record.
step_check: every worked step must be arithmetically valid (expressions) or solution-preserving
(equations), and the trace must end at the recorded answer.
"""

from __future__ import annotations

from fractions import Fraction

from microreason.verify.expr import (
    ExprError,
    evaluate,
    fmt_value,
    parse_equation,
    parse_expression,
    solve_linear,
)
from microreason.verify.symbolic import sympy_expression_value, sympy_linear_solution


def independent_check(kind: str, text: str, prompt: str, final_answer: str) -> str | None:
    if text not in prompt:
        return "math_text_not_in_prompt"
    expected = Fraction(final_answer)
    if fmt_value(expected) != final_answer:
        return "answer_not_canonical"
    try:
        if kind == "expression":
            got = sympy_expression_value(text)
        elif kind == "linear_equation":
            got = sympy_linear_solution(text)
        else:
            return "unknown_kind"
    except (ExprError, ValueError, ZeroDivisionError):
        return "independent_check_failed"
    return None if got == expected else "independent_check_disagrees"


def step_check(kind: str, steps: list[str], final_answer: str) -> str | None:
    if not steps:
        return "no_steps"
    expected = Fraction(final_answer)
    try:
        for step in steps:
            if ": " not in step:
                return "step_missing_label"
            math = step.split(": ", 1)[1]
            if kind == "expression":
                if " = " not in math:
                    return "step_missing_equals"
                lhs, rhs = math.rsplit(" = ", 1)
                if evaluate(parse_expression(lhs)) != evaluate(parse_expression(rhs)):
                    return "step_arithmetic_wrong"
            else:
                lhs, rhs = parse_equation(math)
                if solve_linear(lhs, rhs) != expected:
                    return "step_changes_solution"
        last = steps[-1].split(": ", 1)[1]
        if kind == "expression":
            if evaluate(parse_expression(last.rsplit(" = ", 1)[1])) != expected:
                return "trace_does_not_reach_answer"
        elif last != f"x = {final_answer}":
            return "trace_does_not_reach_answer"
    except ExprError as e:
        return f"step_unparseable:{e}"
    return None
