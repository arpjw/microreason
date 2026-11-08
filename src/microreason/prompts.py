"""The single place where prompts and training sequences are assembled (spec §3).

Show-work SFT format (version sft-v1):

    [bos][user]{prompt} Show your work.[eos][assistant][steps]
    1. ...
    2. ...
    [/steps]
    [answer]{answer}[/answer][eos]

Direct-answer format replaces the instruction with "Give only the answer." and the assistant
turn with "[answer]{answer}[/answer][eos]".

Markers are inserted as IDs, never encoded from text. Loss applies only to assistant tokens
(everything after [assistant], including the final [eos]); prompt and padding labels are -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from microreason.tokenizer import MicroTokenizer

PROMPT_FORMAT_VERSION = "sft-v1"
IGNORE_INDEX = -100

Mode = Literal["show_work", "direct"]
INSTRUCTIONS: dict[str, str] = {
    "show_work": " Show your work.",
    "direct": " Give only the answer.",
}


class SequenceTooLong(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int  # tokens up to and including [assistant]


def user_text(prompt: str, mode: Mode) -> str:
    return prompt + INSTRUCTIONS[mode]


def prompt_ids(tok: MicroTokenizer, prompt: str, mode: Mode) -> list[int]:
    """Prompt ending in [assistant], ready for generation."""
    s = tok.special
    return [
        s["[bos]"],
        s["[user]"],
        *tok.encode(user_text(prompt, mode)),
        s["[eos]"],
        s["[assistant]"],
    ]


def assistant_ids(tok: MicroTokenizer, steps: list[str], answer: str, mode: Mode) -> list[int]:
    s = tok.special
    ids: list[int] = []
    if mode == "show_work":
        body = "\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)) + "\n"
        ids += [s["[steps]"], *tok.encode(body), s["[/steps]"], *tok.encode("\n")]
    ids += [s["[answer]"], *tok.encode(answer), s["[/answer]"], s["[eos]"]]
    return ids


def build_example(
    tok: MicroTokenizer,
    prompt: str,
    steps: list[str],
    answer: str,
    mode: Mode,
    max_len: int | None = None,
) -> Example:
    p = prompt_ids(tok, prompt, mode)
    a = assistant_ids(tok, steps, answer, mode)
    ids = p + a
    if max_len is not None and len(ids) > max_len:
        # Never truncate: a cut answer block would teach a malformed format.
        raise SequenceTooLong(f"{len(ids)} tokens exceeds max_len {max_len}")
    labels = [IGNORE_INDEX] * len(p) + a
    return Example(ids, labels, len(p))


def build_record_example(
    tok: MicroTokenizer, record: dict, mode: Mode, max_len: int | None = None
) -> Example:
    return build_example(
        tok, record["prompt"], record["solution_steps"], record["final_answer"], mode, max_len
    )


def completion_text(tok: MicroTokenizer, generated_ids: list[int]) -> str:
    """Text the verifier sees for IDs generated after [assistant]."""
    return tok.decode(generated_ids)
