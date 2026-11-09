# MicroReason closeout report v1

Date: 2026-11-08. Code revision at writing: `3b51aad` plus this report and the failure analysis.

This report is the spec §11–§12 closeout for the experiment as specified: base pretraining →
math mid-training → SFT → DPO and GRPO branches, with an equal-compute SFT control. Each finding
for H1–H4 is stated together with the evidence behind it.

**Spec §11 asks for four kinds of content, kept apart:**
1. Implemented and tested behaviour.
2. Measured results.
3. Estimates and unrun proposals.
4. Known limitations.

## Findings for H1–H4

| ID | Hypothesis | Finding | Evidence level |
| --- | --- | --- | --- |
| H1 | Math mid-training improves math capability | **Supported** | Val only: v1-math − v0-base = +5.8 pts [+5.2, +6.5] (protocol v2); +23.2 [+22.1, +24.3] at protocol v1. Not in the confirmatory test plan, so not confirmed on test. |
| H2 | SFT improves instruction following and answer format | **Supported** | Val only: under SFT prompts v1-math produces no answer blocks (0%); v2-sft 100% valid, show-work 89.9%, direct 38.2%. Not confirmed on test. **Cost:** v2-sft show-work 89.9% vs v1-math's own plain format 93.9% (different formats, not a paired test). |
| H3 | DPO improves selection or format without a major correctness loss | **Not supported** | Format was already 100% valid, so there was nothing to improve. Correctness fell: val −5.4 [−6.1, −4.7], `test_id` −5.1 [−5.8, −4.4]. Held-out pair discrimination went from 0 to 0.63. |
| H4 | Exact-outcome GRPO improves held-out correctness beyond SFT and equal-cost SFT | **Not supported** (pre-declared rule) | On `test_heldout_template`: v4-grpo-r2 − v2-sft = −0.10 [−0.20, −0.02]; − v2-extra-sft-r2 = −0.13 [−0.25, −0.03]. The rule needed ≥ +2 pts and intervals above 0. **Every model is at the floor there (≤ 0.2%),** so the suite cannot rank methods by generalization. |

**What H4 does and does not show.** H4 is not supported, and the floor makes the held-out test
weak evidence either way. What the data do show:
- **In distribution:** GRPO, in two configurations, made v2-sft slightly worse (v1: −1.5; v2:
  −1.8 on val, −2.3 on `test_id`). Equal-time continued SFT improved it (+4.1 to +4.5).
- **Out of distribution:** neither method changed how the models fail (`reports/failure_analysis_v1.md`).

## Branch status (spec §12)

| Branch | Status | Checkpoint id | Where |
| --- | --- | --- | --- |
| v0-base | completed (protocol v2) | `sha256:d704fc3f…` | `artifacts/p2-v0-base` |
| v1-math | completed (protocol v2) | `sha256:51505ebc…` | `artifacts/p2-v1-math` |
| v2-sft | completed (protocol v2) | `sha256:1ec3ff35…` | `artifacts/p2-v2-sft` |
| v3-dpo | completed (RL v1) | `sha256:4e2c7cff…` | `artifacts/p2-v3-dpo` |
| v2-extra-sft | completed (RL v1, matched to v4-grpo) | `sha256:9e2d0750…` | `artifacts/p2-v2-extra-sft` |
| v4-grpo | completed (RL v1) | `sha256:74c203aa…` | `artifacts/p2-v4-grpo` |
| v4-grpo-r2 | completed (rl-v2, LR 1e-5, selected on val) | `sha256:24a280c8…` | `artifacts/p2-v4-grpo-r2` |
| v4-grpo-v2-lr3e-5 | completed, not selected (unstable) | `sha256:418b46c2…` | `runs/pod_results/tzjzi4mv71cp51/` |
| v2-extra-sft-r2 | completed (rl-v2, matched to v4-grpo-r2) | `sha256:3b85d7fc…` | `artifacts/p2-v2-extra-sft-r2` |
| v4-dpo-grpo (optional) | **not run** | — | — |
| Protocol-v1 v0-base / v1-math / v2-sft | superseded by protocol v2, kept | `55709eec…` / `a6eb6b10…` / `111f81e6…` | `runs/pod_results/` |

All checkpoints are local, not tracked in git, and checksum-verified on pull.

## Measured results (greedy, exact answer; 95% intervals are paired bootstraps over prompt ids)

| Checkpoint | val show-work | val direct | `test_id` | Held-out templates | Extrapolation |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0-base (plain format) | 88.1% | — | — | — | — |
| v1-math (plain format) | 93.9% | — | — | — | — |
| v2-sft | 89.9% | 38.2% | 90.2% | 0.17% | 0.21% |
| v3-dpo | 84.5% | 36.1% | 85.1% | 0.10% | 0.21% |
| v4-grpo (v1) | 88.4% | 36.8% | 88.0% | 0.13% | 0.24% |
| v2-extra-sft (v1) | 93.9% | 41.0% | 93.5% | 0.20% | 0.24% |
| v4-grpo-r2 | 88.1% | 35.5% | 87.8% | 0.07% | 0.26% |
| v2-extra-sft-r2 | 94.4% | 44.0% | 94.3% | 0.20% | 0.21% |

**Suites:**

| Suite | Prompts | sha256 |
| --- | ---: | --- |
| val | 7,000 | `7b2f3679…` |
| `test_id` | 7,000 | `18f17620…` |
| `test_extrapolation` | 7,000 | `e7840ddf…` |
| `test_heldout_template` | 6,000 | `cdcd2109…` |

**Protocol notes:**
- **Test suites:** each was opened once, at 22:57:03 UTC on 2026-11-08, under
  `reports/confirmatory_plan_v1.md`.
- **Valid format:** 99.9–100% for every checkpoint after SFT.
- **Seeds:** one training seed per branch.

## Implemented and tested behaviour

**Code** (`src/microreason/`):
- Data generators and frozen suites with overlap audits.
- Byte-level BPE tokenizer (8,192 tokens).
- A 42.7 M-parameter decoder (RoPE, SwiGLU, tied embeddings); the parameter count matches the
  analytic count in `configs/model_43m.yaml`.
- An exact and symbolic verifier, and a training loop with resume, budget guard and wall-time stop.
- SFT with a masked loss.
- DPO with verifier-labelled pairs.
- GRPO with clipped ratios, k3 KL to the parent, dynamic sampling and immutable rollout records.
- A KV-cache sampler used for rollouts only.
- An evaluation harness with sealed test suites, Wilson intervals and paired bootstraps.

**Tests:** the full local suite passes (CPU, tiny models). The CUDA subset passed on every pod
before training (rl-v2: 51 passed, 1 MPS-only skip). These include analytic sign and gradient
checks for GRPO and DPO, and KV-cache parity checks.

**Infrastructure:**
- RunPod sessions run under per-session budgets, with a self-terminating watchdog.
- Results are pulled and verified locally, and every pod is deleted.

## Cost ledger (GPU time at listed rates; storage excluded; the RunPod billing page is authoritative)

| Session | GPU | Cost | Authorization |
| --- | --- | ---: | --- |
| Phase 5 pilot | RTX 4090 (aborted), RTX 5090 | ≈ $1.03 | $5 cap |
| Stage 1 (v0-base, v1-math, protocol v1) | RTX 4090 | ≈ $0.64 | stage-1 budget |
| Stage 2 (v2-sft, protocol v1) | RTX 4090 | ≈ $0.44 | $2 cap |
| Protocol v2 scaled redo | RTX 5090 | ≈ $2.19 | $8 cap |
| RL v1 (gate, GRPO, extra-SFT, DPO) | RTX 6000 Ada | ≈ $1.75 | $5 cap |
| rl-v2 and confirmatory test | RTX 6000 Ada | ≈ $2.31 | $4 cap |
| **Total** | | **≈ $8.36** | each session within its own approval |

The project's proposed overall cap in the spec was $50.

## Known limitations

- **Synthetic, template-based data.** Fourteen train-visible templates with fixed trace formats.
  The models learned these templates, not general expression parsing: on held-out templates they
  reuse a training template's skeleton in 99.97% of traces.
- **Number-range and tokenizer sensitivity.** Every training operand is a single BPE token (0–99).
  Extrapolation operands are mostly two tokens, and the models miscopy them, typically keeping
  the first token.
- **Out-of-distribution suites at the floor.** They establish that nothing here generalizes, but
  they cannot rank post-SFT methods by generalization.
- **Small model and one seed.** The in-distribution effects are large compared with their
  intervals, but each comes from one training run per branch.
- **Search on val.** GRPO v2 used two learning rates, selected on val. The selection was
  pre-declared, and the test suites played no part in it.
- **Compute matching.** It covers GRPO's training-loop time, not its search cost. GRPO's time was
  82–91% generation, so the SFT control got far more optimizer updates (9× in v1, 28× in v2).
- **Generated steps** are produced by imitating training traces. They should not be read as a
  faithful account of the model's computation; held-out traces visibly are not.
- **Template leakage risk.** It was checked at data build:
  - fingerprint and overlap audits across splits;
  - a scan of 435,019 general-text documents for prompts appearing verbatim. That scan dropped
    93 val and 77 `test_id` candidates, and none from the held-out or extrapolation suites.
- **The test suites are now used.** Any later test evaluation is post-hoc.

## Estimates and unrun proposals

These are not results. Each would need its own protocol and spend approval.

- **Confirming H1 and H2 on test:** the existing suites could evaluate v0-base and v1-math, but
  only as post-hoc analyses; they were not in the confirmatory plan.
- **v3 (a capability-first redesign),** guided by `reports/failure_analysis_v1.md`:
  - **Numbers:** digit-level number tokenization, and operands that span multi-token ranges.
  - **Data:** compositional expression-tree data covering every step type.
  - **Held-out suite:** a new one, frozen before any run, on which the pre-RL model is above 0%.
  - **RL test:** only then re-run the GRPO-vs-control comparison.
  - **Cost:** a rough estimate is a few dollars per session at current rates. It is unmeasured.
- **Second seeds** for the decisive in-distribution comparisons.
