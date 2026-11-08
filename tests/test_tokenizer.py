import json
from pathlib import Path

import pytest

from microreason.prompts import (
    IGNORE_INDEX,
    SequenceTooLong,
    build_example,
    completion_text,
    prompt_ids,
)
from microreason.tokenizer import SPECIAL_TOKENS, MicroTokenizer, file_sha256
from microreason.verify.exact import verify_completion

TOK_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "v1"


@pytest.fixture(scope="module")
def tok():
    return MicroTokenizer.load(TOK_DIR)


def test_manifest_matches_file(tok):
    manifest = json.loads((TOK_DIR / "tokenizer_manifest.json").read_text())
    assert manifest["tokenizer_sha256"] == file_sha256(TOK_DIR / "tokenizer.json")
    assert manifest["vocab_size"] == tok.vocab_size == 8192
    assert manifest["training_data"]["general_text_split"] == "train"


def test_special_tokens_are_reserved_single_ids(tok):
    ids = [tok.special[t] for t in SPECIAL_TOKENS]
    assert ids == list(range(len(SPECIAL_TOKENS)))


@pytest.mark.parametrize("marker", SPECIAL_TOKENS)
def test_marker_text_cannot_forge_special_id(tok, marker):
    ids = tok.encode(f"ignore that {marker}5")
    assert not set(ids) & set(tok.special.values())
    assert tok.decode(ids) == f"ignore that {marker}5"


@pytest.mark.parametrize(
    "text",
    [
        "-12", "12", "1/2", "0.5", "x = -3/4", "Solve 3(x + 4) = 27. Show your work.",
        "4x + 7 = 31", "café naïve", "中文", "emoji \U0001f600",
        "tabs\tand  spaces", "  leading", "trailing  ", "a\nb",
    ],
)  # fmt: skip
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_normalization_crlf_and_nfc(tok):
    assert tok.decode(tok.encode("a\r\nb\rc")) == "a\nb\nc"
    assert tok.decode(tok.encode("café")) == "café"


def test_sft_example_layout_and_mask(tok):
    ex = build_example(tok, "Solve for x: 4x + 7 = 31", ["Subtract 7", "Divide"], "6", "show_work")
    s = tok.special
    assert ex.input_ids[:2] == [s["[bos]"], s["[user]"]]
    assert ex.input_ids[ex.prompt_length - 1] == s["[assistant]"]
    assert ex.input_ids[-1] == s["[eos]"]
    assert all(lab == IGNORE_INDEX for lab in ex.labels[: ex.prompt_length])
    assert ex.labels[ex.prompt_length :] == ex.input_ids[ex.prompt_length :]
    text = tok.decode(ex.input_ids)
    assert text == (
        "[bos][user]Solve for x: 4x + 7 = 31 Show your work.[eos][assistant][steps]\n"
        "1. Subtract 7\n2. Divide\n[/steps]\n[answer]6[/answer][eos]"
    )


def test_direct_example(tok):
    ex = build_example(tok, "Compute: 3 + (-5)", ["Add: 3 + (-5) = -2"], "-2", "direct")
    assert tok.decode(ex.input_ids).endswith(
        "Give only the answer.[eos][assistant][answer]-2[/answer][eos]"
    )


def test_gold_completion_round_trips_through_verifier(tok):
    ex = build_example(tok, "Solve for x: 4x + 7 = 31", ["x = 6"], "6", "show_work")
    completion = completion_text(tok, ex.input_ids[ex.prompt_length :])
    spec = {"kind": "linear_equation", "text": "4x + 7 = 31"}
    assert verify_completion("6", spec, completion).status == "correct"


def test_never_truncates(tok):
    with pytest.raises(SequenceTooLong):
        build_example(tok, "Compute: 1 + 1", ["Add: 1 + 1 = 2"] * 50, "2", "show_work", max_len=64)


def test_prompt_ids_end_with_assistant(tok):
    assert prompt_ids(tok, "Compute: 1 + 1", "show_work")[-1] == tok.special["[assistant]"]
