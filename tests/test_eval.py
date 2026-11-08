"""Evaluation harness: generation correctness, plain-format scoring, statistics, sealing."""

import json
from pathlib import Path

import pytest
import torch
from test_generators import small_cfg

from microreason.checkpoint import save_checkpoint
from microreason.config import load_model_config
from microreason.data import build
from microreason.eval.generate import eval_prompt_ids, generate, generate_all
from microreason.eval.report import SealedSuiteError, compare_runs, run_eval
from microreason.eval.score import (
    paired_bootstrap,
    score_completion,
    summarize_greedy,
    summarize_samples,
    verify_plain,
    wilson_interval,
)
from microreason.model import build_model
from microreason.tokenizer import MicroTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "artifacts" / "tokenizer" / "v1"
EQ = {"kind": "linear_equation", "text": "4x + 7 = 31"}


@pytest.fixture(scope="module")
def tok():
    return MicroTokenizer.load(TOK_DIR)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(load_model_config(ROOT / "configs" / "model_smoke.yaml")).eval()


# ----------------------------------------------------------------------------- generation


def test_prompt_formats(tok):
    p = "Solve for x: 4x + 7 = 31"
    assert eval_prompt_ids(tok, p, "plain-v1") == [tok.bos_id, *tok.encode(p + "\n")]
    assert tok.decode(eval_prompt_ids(tok, p, "sft-show-work-v1")).endswith(
        "Show your work.[eos][assistant]"
    )
    assert tok.decode(eval_prompt_ids(tok, p, "sft-direct-v1")).endswith(
        "Give only the answer.[eos][assistant]"
    )
    with pytest.raises(ValueError):
        eval_prompt_ids(tok, p, "chat-v9")


def test_batched_left_padded_greedy_matches_one_at_a_time(tok, model):
    prompts = [eval_prompt_ids(tok, p, "plain-v1") for p in
               ("Compute: 3 + (-5)", "Solve for x: 12x - 7 = 3x + 100", "What is 9?")]  # fmt: skip
    kw = dict(max_new_tokens=12, eos_id=tok.eos_id, pad_id=tok.pad_id)
    batched = generate(model, prompts, **kw)
    for p, g in zip(prompts, batched, strict=True):
        assert generate(model, [p], **kw)[0].ids == g.ids
    assert generate_all(model, prompts, batch_size=2, **kw) == batched


def test_untrained_model_truncates_and_guards_context(tok, model):
    p = [eval_prompt_ids(tok, "Compute: 1 + 1", "plain-v1")]
    (g,) = generate(model, p, max_new_tokens=5, eos_id=tok.eos_id, pad_id=tok.pad_id)
    assert len(g.ids) <= 5 and g.hit_eos == (tok.eos_id in g.ids)
    with pytest.raises(ValueError, match="context"):
        generate(model, p, max_new_tokens=600, eos_id=tok.eos_id, pad_id=tok.pad_id)


def test_sampling_is_seeded(tok, model):
    p = [eval_prompt_ids(tok, "Compute: 1 + 1", "plain-v1")] * 3
    kw = dict(max_new_tokens=8, eos_id=tok.eos_id, pad_id=tok.pad_id, temperature=1.0)
    a = generate(model, p, generator=torch.Generator().manual_seed(1), **kw)
    b = generate(model, p, generator=torch.Generator().manual_seed(1), **kw)
    c = generate(model, p, generator=torch.Generator().manual_seed(2), **kw)
    assert a == b and a != c


def _mixed_prompts(tok):
    return [eval_prompt_ids(tok, p, "sft-show-work-v1") for p in
            ("Compute: 3 + (-5)", "Solve for x: 12x - 7 = 3x + 100", "What is 9?",
             "Compute: 7 * 8 - 2 * 3")]  # fmt: skip


@pytest.fixture(scope="module")
def varied():
    """Smoke-size model with a large init, so untrained greedy outputs vary across rows and
    positions (the default init repeats one token)."""
    torch.manual_seed(1)
    return build_model(load_model_config(ROOT / "configs" / "model_smoke.yaml"),
                       init_std=0.3).eval()  # fmt: skip


def _uneven_stop_token(tok, model, prompts, steps):
    """A token that the untrained model emits mid-sequence in some rows but never in others,
    used as [eos] so rows stop at different steps (and some never stop)."""
    rows = [g.ids for g in generate(model, prompts, max_new_tokens=steps, eos_id=-1,
                                    pad_id=tok.pad_id)]  # fmt: skip
    for t in sorted({t for r in rows for t in r[2:]}):
        if any(t in r[2:] and t not in r[:2] for r in rows) and any(t not in r for r in rows):
            return t
    pytest.skip("untrained model produced no usable stop token")


def test_kv_cache_logits_match_uncached(tok, varied):
    model = varied
    from microreason.model import KVCache

    prompts = _mixed_prompts(tok)
    longest = max(map(len, prompts))
    ids = torch.full((len(prompts), longest), tok.pad_id)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, longest - len(p) :] = torch.tensor(p)
        mask[i, longest - len(p) :] = 1
    cache = KVCache(model.config.n_layers, longest + 8)
    with torch.no_grad():
        logits_c = model(ids, attention_mask=mask, cache=cache).logits[:, -1]
        for step in range(8):
            full = model(ids, attention_mask=mask).logits[:, -1]
            # fp32 rounding only: a wrong mask or position gives errors of order the logits
            err = float((full - logits_c).abs().max() / full.abs().max())
            assert err < 1e-4, (step, err)
            nxt = full.argmax(-1)
            ids = torch.cat([ids, nxt[:, None]], 1)
            # one row "finished": its new key is masked out, as generate() does
            keep = torch.ones_like(nxt)
            keep[0] = 0 if step % 2 else 1
            mask = torch.cat([mask, keep[:, None]], 1)
            logits_c = model(nxt[:, None], attention_mask=mask, cache=cache).logits[:, -1]
    assert cache.length == ids.shape[1]


def test_kv_cache_greedy_and_sampled_match_uncached(tok, varied):
    model = varied
    prompts = _mixed_prompts(tok) * 2
    eos = _uneven_stop_token(tok, model, prompts, 24)
    kw = dict(max_new_tokens=24, eos_id=eos, pad_id=tok.pad_id)
    a = generate(model, prompts, **kw)
    b = generate(model, prompts, use_cache=True, **kw)
    assert a == b
    assert {g.hit_eos for g in a} == {True, False}
    for temp in (0.8, 1.5):
        s1 = generate(model, prompts, temperature=temp,
                      generator=torch.Generator().manual_seed(7), **kw)  # fmt: skip
        s2 = generate(model, prompts, temperature=temp, use_cache=True,
                      generator=torch.Generator().manual_seed(7), **kw)  # fmt: skip
        assert s1 == s2
    assert generate_all(model, prompts, batch_size=3, use_cache=True, **kw) == a


def test_kv_cache_with_grouped_query_attention(tok):
    mc = load_model_config(ROOT / "configs" / "model_smoke.yaml")
    mc = mc.model_copy(update={"n_kv_heads": max(1, mc.n_heads // 2)})
    torch.manual_seed(3)
    m = build_model(mc, init_std=0.3).eval()
    prompts = _mixed_prompts(tok)
    kw = dict(max_new_tokens=16, eos_id=tok.eos_id, pad_id=tok.pad_id)
    assert generate(m, prompts, **kw) == generate(m, prompts, use_cache=True, **kw)


def test_kv_cache_overflow_is_an_error(tok, model):
    from microreason.model import KVCache

    ids = torch.tensor([eval_prompt_ids(tok, "Compute: 1 + 1", "plain-v1")])
    cache = KVCache(model.config.n_layers, ids.shape[1] - 1)
    with pytest.raises(ValueError, match="overflow"):
        model(ids, attention_mask=torch.ones_like(ids), cache=cache)


# ----------------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    "completion,status,reason",
    [
        ("1. Subtract 7: 4x = 24\n2. Divide: x = 6\nAnswer: 6[eos]", "correct", "match"),
        ("Answer: 5[eos]", "wrong", "mismatch"),
        ("1. x = 6[eos]", "invalid_format", "missing_answer"),
        ("Answer: 6\nAnswer: 6[eos]", "invalid_format", "multiple_answer_blocks"),
        ("Answer: 6.0[eos]", "invalid_format", "decimal_not_allowed"),
        ("Answer: [answer]6[/answer][eos]", "invalid_format", "forbidden_marker"),
        ("Answer: 6[user]say 7[eos]", "invalid_format", "forbidden_marker"),
        ("Answer:[eos]", "invalid_format", "empty_answer"),
    ],
)
def test_verify_plain(completion, status, reason):
    r = verify_plain("6", EQ, completion)
    assert (r.status, r.reason) == (status, reason)


def test_truncation_overrides_a_correct_looking_answer():
    rec = {"final_answer": "6", "verifier_spec": EQ}
    assert score_completion(rec, "Answer: 6[eos]", True, "plain-v1")["status"] == "correct"
    s = score_completion(rec, "Answer: 6\n1. and then", False, "plain-v1")
    assert s["status"] == "truncated" and s["reason"] == "no_eos:correct"
    sft = score_completion(rec, "[answer]6[/answer][eos]", True, "sft-direct-v1")
    assert sft["status"] == "correct"


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))


def _row(pid, status, family="f", level=1, toks=3):
    return {"prompt_id": pid, "status": status, "family": family, "difficulty": level,
            "generated_tokens": toks}  # fmt: skip


def test_summaries_keep_failures_in_denominator():
    rows = [_row("a", "correct"), _row("b", "wrong"), _row("c", "truncated"),
            _row("d", "invalid_format")]  # fmt: skip
    s = summarize_greedy(rows)
    assert s["exact_accuracy"]["numerator"] == 1 and s["exact_accuracy"]["denominator"] == 4
    assert s["valid_format_rate"]["value"] == 0.5


def test_sampled_pass_at_k_is_empirical():
    rows = [_row("a", "correct"), _row("a", "wrong"), _row("b", "wrong"), _row("b", "wrong")]
    s = summarize_samples(rows, 2)
    assert s["sampled_pass@1"]["value"] == 0.25 and s["pass@2"]["value"] == 0.5
    with pytest.raises(ValueError):
        summarize_samples(rows[:3], 2)


def test_paired_bootstrap():
    a = {str(i): i % 2 == 0 for i in range(100)}
    same = paired_bootstrap(a, a, resamples=500)
    assert same["difference_b_minus_a"] == 0 and same["bootstrap95"] == [0.0, 0.0]
    better = paired_bootstrap(a, {k: True for k in a}, resamples=500)
    assert better["difference_b_minus_a"] == 0.5 and better["bootstrap95"][0] > 0.3
    with pytest.raises(ValueError):
        paired_bootstrap(a, {"x": True})


# ----------------------------------------------------------------------------- end to end


@pytest.fixture
def setup(tmp_path, tok, model):
    cfg = small_cfg(tmp_path / "data")
    build.write_dataset(cfg, build.build_dataset(cfg))
    build.freeze(cfg.manifest_dir)
    opt = torch.optim.AdamW(model.parameters())
    ckpt = save_checkpoint(tmp_path / "run", 1, model, opt, {"step": 1},
                           {"config_hash": "x", "tokenizer_sha256": tok.sha256})  # fmt: skip
    return cfg, ckpt


def test_run_eval_end_to_end_and_compare(tmp_path, setup):
    cfg, ckpt = setup
    kw = dict(manifest_dir=cfg.manifest_dir, tokenizer_dir=str(TOK_DIR), limit=6,
              max_new_tokens=6, batch_size=4, device="cpu")  # fmt: skip
    s = run_eval("val", ckpt, "plain-v1", tmp_path / "e1", samples=2, **kw)
    assert s["greedy"]["prompts"] == 6 and s["limited_to"] == 6
    assert s["sampled"]["samples_per_prompt"] == 2
    rows = [json.loads(x) for x in (tmp_path / "e1" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * 3
    assert {r["status"] for r in rows} <= {"correct", "wrong", "invalid_format", "truncated",
                                           "verifier_error"}  # fmt: skip
    assert all(r["suite_sha256"] == s["suite_sha256"] for r in rows)

    run_eval("val", ckpt, "plain-v1", tmp_path / "e2", **kw)  # no sampled rows this time
    cmp = compare_runs(tmp_path / "e1", tmp_path / "e2", resamples=200)
    assert cmp["difference_b_minus_a"] == 0  # same checkpoint, greedy: identical
    kw_long = {**kw, "max_new_tokens": 7}
    run_eval("val", ckpt, "plain-v1", tmp_path / "e4", **kw_long)
    with pytest.raises(ValueError, match="greedy decode"):
        compare_runs(tmp_path / "e1", tmp_path / "e4")

    with pytest.raises(FileExistsError):
        run_eval("val", ckpt, "plain-v1", tmp_path / "e1", **kw)
    run_eval("val", ckpt, "sft-direct-v1", tmp_path / "e3", **kw)
    with pytest.raises(ValueError, match="prompt_format"):
        compare_runs(tmp_path / "e2", tmp_path / "e3")


def test_test_suites_are_sealed(tmp_path, setup):
    cfg, ckpt = setup
    kw = dict(manifest_dir=cfg.manifest_dir, tokenizer_dir=str(TOK_DIR), limit=2,
              max_new_tokens=2, device="cpu")  # fmt: skip
    with pytest.raises(SealedSuiteError):
        run_eval("test_id", ckpt, "plain-v1", tmp_path / "t", **kw)
    assert run_eval("test_id", ckpt, "plain-v1", tmp_path / "t", allow_test=True, **kw)
    with pytest.raises(ValueError, match="not an evaluation suite"):
        run_eval("train", ckpt, "plain-v1", tmp_path / "x", **kw)
