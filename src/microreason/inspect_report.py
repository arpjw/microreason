"""Phase 3 inspection: tokenizer diagnostics, real token counts, model size and memory."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from microreason.prompts import build_record_example
from microreason.tokenizer import SPECIAL_TOKENS, MicroTokenizer, math_plain_text

DIAGNOSTIC_STRINGS = [
    "-12",
    "12",
    "1/2",
    "0.5",
    "x = -3/4",
    "4x + 7 = 31",
    "-30/17",
    "170459",
    "Solve 3(x + 4) = 27. Show your work.",
    "3 + (-5)",
    "(30 + (-16)) * (-77)",
]

# Planned processed-token budgets from the spec (§4, §6).
BASE_TOKENS = 120_000_000
BASE_MATH_FRACTION = 0.30
MID_TOKENS = 40_000_000
MID_MATH_FRACTION = 0.85


def _pieces(tok: MicroTokenizer, text: str) -> list[str]:
    return [tok.decode([i]) for i in tok.encode(text)]


def _dist(values: list[int]) -> dict:
    return dict(sorted(Counter(values).items()))


def number_tokenization(tok: MicroTokenizer) -> dict:
    out = {}
    for label, nums in {
        "0-9": range(10),
        "10-99": range(10, 100),
        "100-999": range(100, 1000),
        "1000-9999 (every 7th)": range(1000, 10000, 7),
    }.items():
        counts = [len(tok.encode(str(n))) for n in nums]
        neg = [len(tok.encode(str(-n))) for n in nums]
        out[label] = {
            "tokens_per_number": _dist(counts),
            "tokens_per_negative_number": _dist(neg),
            "single_token_fraction": round(counts.count(1) / len(counts), 3),
        }
    return out


def _encode_lengths(tok: MicroTokenizer, texts, batch: int = 2048) -> tuple[int, int, int]:
    n_tok = n_chars = n_docs = 0
    buf: list[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) == batch:
            n_tok += sum(map(len, tok.encode_batch(buf)))
            n_chars += sum(map(len, buf))
            n_docs += len(buf)
            buf = []
    if buf:
        n_tok += sum(map(len, tok.encode_batch(buf)))
        n_chars += sum(map(len, buf))
        n_docs += len(buf)
    return n_tok, n_chars, n_docs


def tokenizer_report(tokenizer_dir: str, include_corpus: bool = True) -> dict:
    from microreason.data.build import load_split
    from microreason.data.schema import ALL_SPLITS

    tok = MicroTokenizer.load(tokenizer_dir)
    failures: list[str] = []

    roundtrip = {s: tok.decode(tok.encode(s)) == s for s in DIAGNOSTIC_STRINGS}
    failures += [f"roundtrip failed: {s!r}" for s, ok in roundtrip.items() if not ok]
    forged = [m for m in SPECIAL_TOKENS if set(tok.encode(m)) & set(tok.special.values())]
    failures += [f"marker text encodes to a special id: {m}" for m in forged]
    if sorted(tok.special.values()) != list(range(len(SPECIAL_TOKENS))):
        failures.append("special ids are not 0..8")

    report: dict = {
        "tokenizer_sha256": tok.sha256,
        "vocab_size": tok.vocab_size,
        "special_tokens": tok.special,
        "roundtrip": roundtrip,
        "diagnostic_pieces": {s: _pieces(tok, s) for s in DIAGNOSTIC_STRINGS},
        "marker_text_encodings": {m: len(tok.encode(m)) for m in SPECIAL_TOKENS},
        "number_tokenization": number_tokenization(tok),
    }

    math: dict = {}
    for split in ALL_SPLITS:
        m = json.loads(Path(f"data/manifests/math_{split}_v1.json").read_text())
        recs = [r.model_dump() for r in load_split(m)]
        plain = tok.encode_batch([math_plain_text(r) for r in recs])
        sw = [len(build_record_example(tok, r, "show_work").input_ids) for r in recs]
        da = [len(build_record_example(tok, r, "direct").input_ids) for r in recs]
        math[split] = {
            "records": len(recs),
            "plain_tokens_total": sum(map(len, plain)),
            "chars_per_token_plain": round(
                sum(len(math_plain_text(r)) for r in recs) / sum(map(len, plain)), 3
            ),
            "sft_show_work_tokens": {"mean": round(statistics.mean(sw), 1), "max": max(sw),
                                     "total": sum(sw)},
            "sft_direct_tokens": {"mean": round(statistics.mean(da), 1), "max": max(da)},
        }  # fmt: skip
    report["math"] = math

    if include_corpus:
        from microreason.data.text_split import iter_split_texts

        general = {}
        for split in ("train", "val"):
            t0 = time.time()
            n_tok, n_chars, n_docs = _encode_lengths(
                tok, iter_split_texts("configs/data_sources.yaml", "data/raw", split)
            )
            general[split] = {
                "documents": n_docs,
                "tokens": n_tok,
                "chars": n_chars,
                "chars_per_token": round(n_chars / n_tok, 3),
                "encode_seconds": round(time.time() - t0, 1),
            }
        report["general_text"] = general

        text_avail = general["train"]["tokens"]
        math_avail = math["train"]["plain_tokens_total"]
        need = {
            "base_general_tokens": int(BASE_TOKENS * (1 - BASE_MATH_FRACTION)),
            "base_math_tokens": int(BASE_TOKENS * BASE_MATH_FRACTION),
            "mid_general_tokens": int(MID_TOKENS * (1 - MID_MATH_FRACTION)),
            "mid_math_tokens": int(MID_TOKENS * MID_MATH_FRACTION),
        }
        report["budget_check"] = {
            "needed": need,
            "available_unique_general_train_tokens": text_avail,
            "available_unique_math_train_tokens_plain": math_avail,
            "general_epochs_needed": round(
                (need["base_general_tokens"] + need["mid_general_tokens"]) / text_avail, 3
            ),
            "math_epochs_needed_with_current_train_split": round(
                (need["base_math_tokens"] + need["mid_math_tokens"]) / math_avail, 1
            ),
        }

    report["failures"] = failures
    report["passed"] = not failures
    return report


# ----------------------------------------------------------------------------- model


def _activation_bytes(cfg, batch: int, seq: int, bytes_per: int = 2) -> int:
    """Rough per-step activation memory for training without recomputation.

    Per layer: inputs to the two norms, q/k/v/o, attention output (SDPA keeps no T x T matrix),
    and the three SwiGLU tensors, in bf16. Logits are kept in bf16 plus an fp32 copy for the
    loss. An estimate only; the GPU pilot must measure the real peak.
    """
    d, m, t = cfg.d_model, cfg.mlp_hidden, batch * seq
    per_layer = t * (d * 2 + d * 4 + d + 3 * m) * bytes_per
    logits = t * cfg.vocab_size * (bytes_per + 4)
    return cfg.n_layers * per_layer + logits + t * d * bytes_per


def model_report(config_path: str, measure_mps: bool = True) -> dict:
    import torch

    from microreason.config import load_model_config
    from microreason.model import build_model, count_parameters

    mc = load_model_config(config_path)
    model = build_model(mc)
    n = count_parameters(model)
    by_module = Counter()
    for name, p in model.named_parameters():
        key = name.split(".")[0] if not name.startswith("blocks.") else "blocks"
        by_module[key] += p.numel()

    report: dict = {
        "name": mc.name,
        "parameter_count_analytic": mc.parameter_count(),
        "parameter_count_instantiated": n,
        "counts_match": n == mc.parameter_count(),
        "parameters_by_module": dict(by_module),
        "memory_estimates_gib": {
            "weights_fp32": round(n * 4 / 2**30, 3),
            "training_state_fp32_weights_grads_adamw": round(n * 16 / 2**30, 3),
            "activations_bf16_by_micro_batch_at_seq_1024": {
                str(b): round(_activation_bytes(mc, b, 1024) / 2**30, 2) for b in (4, 8, 16, 32, 64)
            },
            "note": "Estimates only. Measure real peak memory in the authorized GPU pilot.",
        },
    }

    if measure_mps and torch.backends.mps.is_available():
        dev = torch.device("mps")
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        b, t = 4, 1024
        x = torch.randint(9, mc.vocab_size, (b, t), device=dev)
        for _ in range(2):  # warmup
            model(x, labels=x).loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        steps = 5
        t0 = time.time()
        for _ in range(steps):
            model(x, labels=x).loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        dt = (time.time() - t0) / steps
        report["mps_measurement"] = {
            "device": "Apple MPS",
            "precision": "fp32",
            "micro_batch": b,
            "seq_len": t,
            "seconds_per_step": round(dt, 3),
            "tokens_per_second": int(b * t / dt),
            "driver_allocated_gib": round(torch.mps.driver_allocated_memory() / 2**30, 2),
            "note": "Local fp32 measurement; not a forecast of rented-GPU throughput.",
        }
    return report
