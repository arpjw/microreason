# Protocol v3 Phase B: data v3, tokenizer v2, v0-base → v1-math → v2-sft, capability gate

Date: 2026-11-09 UTC (the night of 2026-11-08 local). Protocol: `reports/protocol_v3.md`, frozen
before this session (code revision `83921bc`).

**Session:** RunPod pod `pppweshg0o8h9t`, RTX 6000 Ada (Secure, US-WA-1), listed $0.99/h.
- **Bounds:** $6 and 4.5 h (`configs/budget_v3_phase_b.yaml`, owner-approved).
- **Cost:** about 2 h 48 min, **≈ $2.76**. The pod is deleted.
- **Results:** in `reports/v3_phase_b_results/`.

**Scope:**
- Val and held-out-*val* only. Every v3 test suite stayed sealed: the session script refuses any
  `test_*` evaluation.
- One seed per stage.

## Headline

**The capability gate passed by a wide margin. Phase C is unblocked, subject to its own spend
approval.**

| v2-sft on held-out-val shapes (greedy show-work) | Result | Gate threshold |
| --- | ---: | ---: |
| Accuracy | **70.3%** (4,215 / 6,000) | 10% |
| Sampled pass@4 | **74.4%** | 20% |

For comparison, v1's held-out templates were solved **0.17%** of the time.

**Unseen shapes cost nothing.** On problems whose shape never appears in training, v2-sft is
about as accurate as on training shapes of the same size:

| Operators (v2-sft, show-work greedy) | In-distribution val | Held-out-val shapes |
| --- | ---: | ---: |
| 1 | 94.7% | — (never held out) |
| 2 | 79.6% | — (never held out) |
| 3 | 63.7% | 65.2% |
| 4 | 50.5% | 50.4% |
| 5 | 36.6% | 39.4% |
| Linear equations | 95.8% | 94.8% |

**Reading:**
- **Shape novelty:** with digit tokens and tree-generated data, the model shows no measurable
  cost from shape novelty. v1's failure (template substitution, 99.97% of held-out traces) is
  gone.
- **Depth:** what limits accuracy is **depth**, which falls from 95% at one operator to about
  37% at five.
- **The declared risk, realised:** a held-out shape is a new combination of parts seen in
  training, and it is handled as well as a training shape.
- **What this means for H4:** the held-out-shape test now measures problem-solving on deep
  expressions, not adaptation to novelty.

## Data v3 and tokenizer v2 (built on the pod; frozen before any training)

**Build:**

| Split | Records |
| --- | ---: |
| val | 7,000 |
| `test_id` | 7,000 |
| `test_extrapolation` | 7,000 |
| `val_heldout_shape` | 6,000 |
| `test_heldout_shape` | 6,000 |
| SFT pool | 100,000 |

**Checks:**
- **Corpus scan:** 435,019 general-text documents scanned. It dropped 60 val and 69 `test_id`
  candidates, and none from the held-out or extrapolation suites.
- **Audit:** passed, including regenerating records from their seeds. The frozen file is
  `data/manifests/math_eval_frozen_v3.json`; `test_heldout_shape` is sha256 `a370443a…`.

**Stream:** 6,000,000 records, 37,560 distinct train-visible shapes (sha256 `d359e601…`).
- **Shard a:** 3.60 M records, 346 M tokens, for base pretraining.
- **Shard b:** 2.40 M records for mid-training: easy bucket 41 M tokens, hard bucket 189 M tokens.

**Tokenizer v2:** sha256 `c6266b10…`, now committed. It uses `Digits(individual_digits=True)`,
then byte-level BPE; 24.8% of its training characters are math.

**Setup and data took about 25 minutes of billed GPU time.**

## Training

| Stage | Steps | Tokens or examples | Wall time | Final val loss |
| --- | ---: | --- | ---: | --- |
| v0-base | 15,259 | 1.0 B tokens (70/30 general/math) | 98.8 min (≈170 k tok/s) | general 2.353, math 0.356 |
| v1-math | 6,104 | 200 M tokens (85/15, curriculum) | 20.4 min | general 2.428, math 0.357 |
| v2-sft | 1,563 | 100,032 examples (2 epochs of 50 k) | 2.3 min | show-work 0.0176, direct 1.053 |

The RTX 6000 Ada ran at about 170 k tokens/s, against about 265 k on the v2 RTX 5090. Base
pretraining therefore took 99 minutes rather than 62.

## Accuracy (greedy, `eval-v2-cached`, 320 new tokens)

| Checkpoint | Format | Val | Held-out-val | Notes |
| --- | --- | ---: | ---: | --- |
| v0-base | plain | 75.6% | — | 100% valid |
| v1-math | plain | **80.8%** | 73.5% | 100% valid |
| v2-sft | show-work | 78.1% (pass@1 76.3, pass@4 81.5) | **70.3%** (pass@1 67.6, pass@4 74.4) | 99.99% valid |
| v2-sft | direct | 11.4% | 1.7% | Near zero beyond one operator |
| v1-math | SFT show-work prompt | 0.0% | — | Never produces an answer block; H2 baseline |

**Paired comparisons on val** (bootstrap, 10,000 resamples):
- **H1:** v1-math − v0-base = **+5.19 [+4.54, +5.83]** (plain format). Supported on val,
  matching protocol v2's +5.8.
- **H2:** v2-sft − v1-math, under SFT prompts = **+78.1 [+77.1, +79.0]**. Supported on val.
- **SFT's cost:** v2-sft in show-work is 2.8 points below v1-math in plain on val, and 3.2 points
  below on held-out-val. These are different formats, so the comparison is not paired. v2 showed
  the same pattern.

**By kind on val** (v2-sft, show-work):
- **Expressions:** 94.7%, 79.6%, 63.7%, 50.5% and 36.6% from 1 to 5 operators.
- **Equations:** 95.8%.

v3 problems are about 4× longer than v1's (76 generated tokens on average in val), so v3
accuracy cannot be compared with v1/v2 numbers.

## What this means for Phase C (not run)

- **Headroom:** RL now has room to improve, and a non-floor held-out test.
  - On held-out-val, v2-sft's sampled pass@1 is 67.6% and pass@4 74.4%.
  - Deep expressions (4–5 operators) sit at 40–50%, where groups should often mix correct and
    wrong answers.
- **Protocol as frozen:** Phase C runs the reward-variance gate on a v3 RL pool, then 2 seeds
  each of v4-grpo (LR 1e-5, dynamic sampling, group 8, 256-token rollouts) and its time-matched
  v2-extra-sft.
- **Confirmatory test:** one evaluation on `test_heldout_shape` under a confirmatory plan v2,
  committed before the test is opened.
- **Still to write:** the v3 RL pool (a separate seed stream with partitions), the GRPO v3 and
  extra-SFT v3 configs, the confirmatory plan v2, and the Phase C session script.
- **Spend:** Phase C needs its own approval. The protocol estimate is **≈ $2–3**. The slower
  GPU and longer rollouts (about 3× v1's tokens) may push it toward **$3–4**; I will re-estimate
  once the Phase C code exists.

## Incidents

- The launch SSH call for setup hung, as in earlier sessions, and was stopped locally; the remote
  job was unaffected. The session launch used `timeout` and `disown` and did not hang.
- The pull listed `math_eval_frozen_v3.json` twice, so tar printed a "hardlink to itself"
  warning. The file arrived intact and is identical to the pod's copy.
- No idle billing beyond setup: the watchers used log markers.

## Artifacts

- **Checkpoints** (local, not tracked; checksum-verified): `artifacts/p3-v0-base`,
  `artifacts/p3-v1-math` and `artifacts/p3-v2-sft`. v2-sft's checkpoint id is `sha256:8c2cf6b9…`.
- **Per-prompt eval rows:** `runs/pod_results/pppweshg0o8h9t/runs/p3/eval/*/rows.jsonl`.
- **Tracked in git:** the frozen v3 manifests and tokenizer v2. Phase C regenerates the suite
  data files from the same seeds, and the build refuses to write anything that differs from the
  frozen hashes.
