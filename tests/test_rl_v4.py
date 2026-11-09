# ruff: noqa: F811  (pytest fixtures imported from test_rl)
"""Protocol v4 RL options: step-prefix credit, advantage modes, Dr. GRPO token-constant loss,
clip-higher, micro-batch accumulation, the pass-rate filter, the RFT objective, the filter-yield
gate, and an end-to-end check that the real sample -> score -> update loop raises reward."""

import json
from pathlib import Path

import pytest
import torch
from test_rl import SMOKE, TOK, model_cfg, parent, records, tok  # noqa: F401  (fixtures)

from microreason.rl.config import GRPOConfig, ResolvedRL
from microreason.rl.gate import decide, filter_yield
from microreason.rl.grpo import GRPOTrainer, group_advantages, grpo_loss
from microreason.rl.rollout import Completion, passes_filter, step_prefix_credit

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------- step credit

GOLD = ["Add: 1 + 2 = 3", "Multiply: 3 * 4 = 12"]


def _trace(*steps: str, tail: str = "[answer]12[/answer]") -> str:
    body = "\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) + "\n"
    return f"[steps]{body}[/steps]\n{tail}"


def test_step_prefix_credit_counts_exact_prefix_only():
    assert step_prefix_credit(_trace(*GOLD), GOLD) == 1.0
    assert step_prefix_credit(_trace(GOLD[0], "Multiply: 3 * 4 = 13"), GOLD) == 0.5
    # an error in step 1 earns nothing even if step 2 is right
    assert step_prefix_credit(_trace("Add: 1 + 2 = 4", GOLD[1]), GOLD) == 0.0
    # padding with repeated correct steps cannot exceed the gold count
    assert step_prefix_credit(_trace(GOLD[0], GOLD[0], GOLD[1]), GOLD) == 0.5
    assert step_prefix_credit(_trace(*GOLD, GOLD[1]), GOLD) == 1.0
    assert step_prefix_credit("[answer]12[/answer]", GOLD) == 0.0
    assert step_prefix_credit(_trace(*GOLD), []) == 0.0
    # wrong numbering is a mismatch
    assert step_prefix_credit("[steps]\n2. " + GOLD[0] + "\n[/steps]", GOLD) == 0.0


def test_shaped_reward_keeps_every_correct_answer_above_every_wrong_one():
    right_bad_steps = Completion([1], True, "correct", "match", 1.0 + 0.2 * 0.0, 0.0)
    wrong_good_steps = Completion([1], True, "wrong", "mismatch", 0.0 + 0.2 * 1.0, 1.0)
    assert right_bad_steps.correct and not wrong_good_steps.correct
    assert right_bad_steps.reward > wrong_good_steps.reward
    with pytest.raises(ValueError):
        GRPOConfig(model="m", step_reward_weight=1.0)


# ----------------------------------------------------------------------------- advantages


def test_advantage_modes():
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    g = group_advantages(r, mode="group_std")
    m = group_advantages(r, mode="group_mean")
    assert torch.allclose(m[0], torch.tensor([0.75, -0.25, -0.25, -0.25]))
    assert torch.all(m[1] == 0) and torch.all(g[1] == 0)  # zero-variance group: no signal
    # group std inflates the 1-of-4 group relative to the balanced one; group_mean does not
    assert g[0, 0] / g[2, 0] > m[0, 0] / m[2, 0]
    rows = torch.tensor([True, False, True])
    b = group_advantages(r, mode="batch_std", batch_rows=rows)
    bstd = r[rows].std(unbiased=False)
    assert torch.allclose(b[0], m[0] / (bstd + 1e-6)) and torch.all(b[1] == 0)
    assert torch.equal(group_advantages(r, mode="reward"), r)  # RFT: A = r, no baseline
    with pytest.raises(ValueError):
        group_advantages(r, mode="nope")


# ----------------------------------------------------------------------------- loss


def _lp(b=3, t=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return -torch.rand(b, t, generator=g) * 3


def test_token_constant_loss_has_no_length_normalisation():
    new = _lp().requires_grad_(True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    adv = torch.tensor([1.0, 1.0, 1.0])
    loss, _ = grpo_loss(new, new.detach(), new.detach(), mask, adv, 0.2, 0.0,
                        aggregation="token_constant", max_len=8)  # fmt: skip
    loss.backward()
    # on-policy: d loss / d lp = -A / (B * max_len) for every unmasked token, regardless of length
    expect = torch.where(mask, torch.full_like(new, -1 / (3 * 8)), torch.zeros_like(new))
    assert torch.allclose(new.grad, expect)
    with pytest.raises(ValueError):
        grpo_loss(new, new.detach(), new.detach(), mask, adv, 0.2, 0.0,
                  aggregation="token_constant")  # fmt: skip


def test_micro_batches_accumulate_to_the_full_batch_gradient():
    new = _lp(4, 6, 1).requires_grad_(True)
    old = _lp(4, 6, 2)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[1, 3:] = False
    adv = torch.tensor([0.5, -1.0, 2.0, 0.0])
    for agg in ("sequence_mean", "token_constant"):
        full, _ = grpo_loss(new, old, old, mask, adv, 0.2, 0.01, aggregation=agg, max_len=6)
        (gf,) = torch.autograd.grad(full, new)
        parts = 0
        for idx in (torch.tensor([0, 1]), torch.tensor([2, 3])):
            lpart, _ = grpo_loss(new[idx], old[idx], old[idx], mask[idx], adv[idx], 0.2, 0.01,
                                 aggregation=agg, max_len=6, denom=4)  # fmt: skip
            parts = parts + lpart
        (gp,) = torch.autograd.grad(parts, new)
        assert torch.allclose(full, parts) and torch.allclose(gf, gp, atol=1e-7)


def test_clip_higher_is_asymmetric():
    old = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    pos = torch.tensor([1.0])
    # ratio 1.25: clipped by symmetric 0.2, free under eps_high 0.28
    new = torch.full((1, 1), float(torch.log(torch.tensor(1.25)))).requires_grad_(True)
    lsym, _ = grpo_loss(new, old, old, mask, pos, 0.2, 0.0)
    (gsym,) = torch.autograd.grad(lsym, new)
    lhi, st = grpo_loss(new, old, old, mask, pos, 0.2, 0.0, clip_eps_high=0.28)
    (ghi,) = torch.autograd.grad(lhi, new)
    assert gsym.abs().sum() == 0 and ghi.abs().sum() > 0 and st["clip_fraction"] == 0
    # the low side still clips at 1 - 0.2 for negative advantages
    low = torch.full((1, 1), float(torch.log(torch.tensor(0.7)))).requires_grad_(True)
    ll, _ = grpo_loss(low, old, old, mask, torch.tensor([-1.0]), 0.2, 0.0, clip_eps_high=0.28)
    (gl,) = torch.autograd.grad(ll, low)
    assert gl.abs().sum() == 0


# ----------------------------------------------------------------------------- filter / gate


def _g(*rewards):
    return [Completion([1], True, "correct" if r >= 1 else "wrong", "", r) for r in rewards]


def test_filter_bounds_and_equal_rewards():
    assert passes_filter(_g(1, 1, 0, 0), 2, 3)
    assert not passes_filter(_g(1, 0, 0, 0), 2, 3)
    assert not passes_filter(_g(1, 1, 1, 1), 2, 3)
    # step-reward arm: 0 correct allowed if step credit differs, never if rewards are all equal
    assert passes_filter(_g(0.1, 0.0, 0.2, 0.0), 0, 3)
    assert not passes_filter(_g(0.0, 0.0, 0.0, 0.0), 0, 3)
    cfg = GRPOConfig(model="m", sampling={"group_size": 4}, filter_min_correct=2)
    assert cfg.filter_bounds() == (2, 3)
    assert filter_yield([_g(1, 1, 0, 0), _g(1, 0, 0, 0)], cfg) == 0.5
    with pytest.raises(ValueError):
        GRPOConfig(model="m", sampling={"group_size": 4}, filter_min_correct=3,
                   filter_max_correct=2)  # fmt: skip
    with pytest.raises(ValueError):
        GRPOConfig(model="m", clip_eps=0.2, clip_eps_high=0.1)


def test_v4_gate_bounds_sampling_cost_not_mixed_fraction():
    cfg = GRPOConfig(model="m", gate={"min_mixed_group_fraction": None, "min_filter_yield": 0.125})
    ok = {"mixed_group_fraction": 0.05, "malformed_fraction": 0.0, "pass_at_1": 0.6,
          "filter_yield": 0.2}  # fmt: skip
    assert decide(ok, cfg) == (True, [])
    passed, why = decide({**ok, "filter_yield": 0.1}, cfg)
    assert not passed and "filter_yield" in why[0]
    # v1-v3 configs keep the old rule
    assert not decide({**ok, "mixed_group_fraction": 0.1}, GRPOConfig(model="m"))[0]


def test_shipped_v4_configs_declare_the_registered_recipe():
    from microreason.rl.config import load_rl_config

    seeds = [load_rl_config(ROOT / f"configs/grpo_v4_seed{n}.yaml").stage for n in (1, 2, 3)]
    for c in seeds:
        assert (c.objective, c.advantage, c.loss_aggregation) == ("grpo", "batch_std",
                                                                   "token_constant")  # fmt: skip
        assert c.minibatches_per_step == 1 and c.kl_beta == 1e-3 and c.step_reward_weight == 0
        assert c.sampling.group_size == 16 and c.prompts_per_step == 64 and c.steps == 150
        assert c.filter_bounds() == (2, 14) and c.gate.min_mixed_group_fraction is None
    assert len({c.seed for c in seeds}) == 3
    rft = load_rl_config(ROOT / "configs/rft_v4_seed1.yaml").stage
    step = load_rl_config(ROOT / "configs/grpo_step_v4_seed1.yaml").stage
    # secondary arms differ from seed 1 only where declared
    s1 = seeds[0].model_dump()
    diff = lambda c: {k for k, v in c.model_dump().items() if s1[k] != v}  # noqa: E731
    assert diff(rft) == {"objective", "kl_beta"}
    assert diff(step) == {"step_reward_weight", "filter_min_correct"}


# ----------------------------------------------------------------------------- end to end


def _v4_cfg(parent_path, **kw):
    base = dict(model=str(SMOKE), tokenizer=str(TOK), parent=str(parent_path), precision="fp32",
                sampling={"group_size": 4, "max_new_tokens": 3, "generation_batch_size": 16,
                          "kv_cache": True},
                prompts_per_step=2, steps=2, warmup_steps=0, checkpoint_every_steps=10,
                log_every_steps=1, dynamic_sampling=True, sample_chunk_prompts=4,
                max_sample_rounds=4, minibatches_per_step=1, advantage="batch_std",
                loss_aggregation="token_constant", clip_eps_high=0.28, kl_beta=1e-3,
                filter_min_correct=1, filter_max_correct=3, mask_truncated=True,
                update_micro_batch=3)  # fmt: skip
    base.update(kw)
    return GRPOConfig(**base)


def _parity_reward(monkeypatch, target_parity=0):
    """Stub reward: 1 iff the first sampled token id has the target parity (a learnable rule)."""
    real = GRPOTrainer._sample

    def fake(self, recs):
        groups = real(self, recs)
        for g in groups:
            for c in g:
                ok = bool(c.ids) and c.ids[0] % 2 == target_parity
                c.reward, c.status = (1.0, "correct") if ok else (0.0, "wrong")
        return groups

    monkeypatch.setattr(GRPOTrainer, "_sample", fake)


@pytest.mark.parametrize("objective", ["grpo", "rft"])
def test_v4_tiny_run_completes_and_records_v4_metrics(tmp_path, model_cfg, parent, records,
                                                      monkeypatch, objective):  # fmt: skip
    _parity_reward(monkeypatch)
    cfg = _v4_cfg(parent, objective=objective)
    run = tmp_path / objective
    res = GRPOTrainer(ResolvedRL(stage=cfg, model=model_cfg), run, "cpu", pool=records).train()
    assert res["status"] == "completed" and res["updates"] == 2  # one update per step
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["event"] == "train"]
    assert all({"filter_yield", "pass_rate", "sampled_token_nll"} <= set(r) for r in train)
    man = json.loads((run / "run_manifest.json").read_text())
    desc = man["provenance"]["objective"]
    assert ("RFT" in desc) == (objective == "rft") and "token_constant" in desc
    used = [json.loads(x) for x in (run / "rollouts.jsonl").read_text().splitlines()]
    used = [r for r in used if r["used_in_loss"]]
    if objective == "rft":  # A = reward: wrong samples carry zero advantage
        assert all(r["advantage"] == r["reward"] for r in used)


def test_real_loop_raises_reward_on_a_learnable_rule(tmp_path, model_cfg, parent, records,
                                                     monkeypatch):  # fmt: skip
    """End-to-end sign check of sample -> score -> advantage -> update (the rl-v2 run left
    training reward flat). With a learnable stub reward, pass rate must rise clearly."""
    _parity_reward(monkeypatch, target_parity=0)
    cfg = _v4_cfg(parent, sampling={"group_size": 8, "max_new_tokens": 1,
                                    "generation_batch_size": 64, "kv_cache": True},
                  prompts_per_step=4, steps=40, lr=3e-3, kl_beta=0.0, dynamic_sampling=False,
                  filter_min_correct=None, filter_max_correct=None,
                  update_micro_batch=8)  # fmt: skip
    run = tmp_path / "learn"
    tr = GRPOTrainer(ResolvedRL(stage=cfg, model=model_cfg), run, "cpu", pool=records * 30)
    tr.train()
    rates = [json.loads(x)["pass_rate"] for x in (run / "metrics.jsonl").read_text().splitlines()
             if json.loads(x)["event"] == "train"]  # fmt: skip
    first, last = sum(rates[:5]) / 5, sum(rates[-5:]) / 5
    assert last > first + 0.25 and last >= 0.9, (first, last)


# ----------------------------------------------------------------------------- v4 data


def test_v4_pools_use_declared_keep_and_distinct_ids(tmp_path):
    from collections import Counter

    import yaml
    from test_data_v3 import _cfg

    from microreason.data import build_v3 as b3
    from microreason.prompts import assistant_ids
    from microreason.rl.rollout import score
    from microreason.tokenizer import MicroTokenizer

    cfg = _cfg(tmp_path)
    b3.write_v3(cfg, b3.build_v3(cfg))
    b3.freeze_v3(cfg.manifest_dir)
    ids, recs_by = {}, {}
    for name in ("rl_pool_v4", "sft_pool_v4"):
        raw = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        raw.update(records=1500, part_size=1000, workers=1,
                   out_path=str(tmp_path / name / "pool.jsonl"),
                   manifest_path=str(tmp_path / "manifests" / f"{name}.json"))  # fmt: skip
        pool = b3.RLPoolV3.model_validate(raw)
        m = b3.build_rl_pool_v3(cfg.model_copy(update={"rl_pool": pool}))
        assert m["family_keep_probability"] == raw["family_keep"]
        recs = [json.loads(x) for x in open(m["file"])]
        assert all(r["id"].startswith(raw["id_prefix"] + "_") for r in recs)
        ids[name] = {r["id"] for r in recs}
        recs_by[name] = recs
    assert not ids["rl_pool_v4"] & ids["sft_pool_v4"]
    fam = Counter(r["family"] for r in recs_by["rl_pool_v4"])
    n = sum(fam.values())
    deep = (fam["expression_3op"] + fam["expression_4op"] + fam["expression_5op"]) / n
    assert deep > 0.6 and fam["linear_equation"] / n < 0.25  # declared target: ~70% / 15%
    natural = Counter(r["family"] for r in recs_by["sft_pool_v4"])
    assert natural["linear_equation"] / sum(natural.values()) > 0.3  # unweighted mix
    # the step credit works on real records rendered in the exact show-work format
    tok = MicroTokenizer.load(ROOT / "artifacts" / "tokenizer" / "v2")
    for r in recs_by["rl_pool_v4"][:50]:
        gold = assistant_ids(tok, r["solution_steps"], r["final_answer"], "show_work")
        c = score(r, gold, True, tok, step_reward_weight=0.2)
        assert c.correct and c.step_credit == 1.0 and abs(c.reward - 1.2) < 1e-9
        wrong_steps = ["Add: 0 + 0 = 1", *r["solution_steps"][1:]]
        bad = assistant_ids(tok, wrong_steps, "987654", "show_work")
        c2 = score(r, bad, True, tok, step_reward_weight=0.2)
        assert c2.step_credit == 0.0 and c2.reward < 1.0


def test_parallel_sft_example_building_matches_sequential(records):
    from microreason.tokenizer import MicroTokenizer
    from microreason.train.sft import build_examples

    tk = MicroTokenizer.load(TOK)
    seq = build_examples(tk, str(TOK), records, 0.75, 384, workers=1)
    par = build_examples(tk, str(TOK), records, 0.75, 384, workers=2, parallel_min=0, chunk=7)
    assert [i for i, _ in par] == [r["id"] for r in records]
    assert [(i, e.input_ids, e.labels) for i, e in par] == [
        (i, e.input_ids, e.labels) for i, e in seq
    ]


def test_replication_suite_is_fresh_frozen_and_reproducible(tmp_path):
    from test_data_v3 import _cfg

    from microreason.data import build_v3 as b3
    from microreason.eval.report import load_frozen_suite

    cfg = _cfg(tmp_path)
    b3.write_v3(cfg, b3.build_v3(cfg))
    b3.freeze_v3(cfg.manifest_dir)
    p = b3.build_replication_suite(cfg, root_seed=99, n=50)
    m = json.loads(p.read_text())
    assert m["version"] == "v4x" and m["record_count"] == 50
    recs, sha = load_frozen_suite("test_extrapolation", cfg.manifest_dir, "v4x")
    assert sha == m["sha256"] and all(r["id"].startswith("v4x_test_extrapolation_") for r in recs)
    assert all(r["number_range"] == "extrapolation" for r in recs)
    old = json.loads(b3.manifest_path(cfg.manifest_dir, "v3", "test_extrapolation").read_text())
    v3 = set()
    for split in b3.V3_EVAL_SPLITS:
        mm = json.loads(b3.manifest_path(cfg.manifest_dir, "v3", split).read_text())
        v3 |= {r.semantic_fingerprint for r in b3.load_split_v3(mm)}
    assert not {r["semantic_fingerprint"] for r in recs} & v3 and old["sha256"] != sha
    assert b3.build_replication_suite(cfg, root_seed=99, n=50) == p  # rebuild matches frozen
    with pytest.raises(ValueError, match="frozen hash"):
        b3.build_replication_suite(cfg, root_seed=100, n=50)
