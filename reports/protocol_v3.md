# Protocol v3: a capability-first redesign, then H4 again

Date: 2026-11-08.

**Status:**
- **Design approved:** the owner approved this design and the recommended choices in chat on
  2026-11-08 ("Approve, start Phase A"): in-distribution operands 0–999, extrapolation 4–5
  digits, DPO dropped, 2 seeds for the RL branches.
- **Phase A is complete:** local code and tests on tiny models, CPU only.
- **The settings below are frozen** for Phase B, as amended in "Phase A results and frozen
  settings" at the end. **No pod has run.**
- **Spend:** Phases B and C each still need their own spend approval.

## Why v3

The v1 experiment is closed (`reports/closeout_v1.md`). H4 could not be tested meaningfully,
because every model scored at most 0.2% on held-out templates and on number extrapolation.
`reports/failure_analysis_v1.md` traced that floor to the data and the tokenizer, not to the
post-SFT methods:

1. **Number copying.** Every training operand was a single BPE token, and the models could not
   copy multi-token numbers from the prompt. 82% of extrapolation copy errors kept only the first
   token.
2. **Template substitution.** The 14 fixed templates taught 14 fixed trace scripts. On held-out
   templates the models reused a training skeleton 99.97% of the time, binding operands by
   position.
3. **Unreachable step types.** One held-out template needed `Expand` and `Combine` steps that
   appear in no training trace.

v3 asks two questions, in order:
- **Q1 (capability):** with digit-level numbers and compositional training data, does the
  pre-RL model solve a measurable share of problems with held-out structure?
- **Q2 (H4, only if Q1 is yes):** does exact-outcome GRPO raise held-out correctness beyond SFT
  and equal-compute continued SFT?

## What changes

### 1. Tokenizer v2: one token per digit

- **Change:** add `pre_tokenizers.Digits(individual_digits=True)` before the existing byte-level
  pre-tokenizer, then retrain BPE on the same training sources with the same 8,192-token
  vocabulary. Every number becomes a sequence of digit tokens, so `167` and `16` share their
  leading tokens and nothing depends on BPE merges.
- **Cost:** numbers take more tokens. Phase A measures the effect on example lengths and
  processed-token budgets before any configs are frozen.
- **Consequence:** the tokenizer changes, so **every stage is retrained from random
  initialization.** No v1 or v2 checkpoint can be reused. The checkpoints are named `p3-*`.

### 2. Math generator v3: compositional expression trees

The 14 fixed templates are replaced by a seeded grammar.

**Expressions** are random binary trees over `+ − × ÷` with parentheses where precedence needs
them:
- 2 to 6 operands, depth 1 to 4;
- exact rational results, no division by zero.

**Linear equations** are `L(x) = R(x)`. Each side is a tree in which `x` appears linearly, for
example `a(x + b) + c = dx + e` or `(x − a)·b − c = d`.

**Canonical traces** are deterministic and use one fixed step vocabulary:

| Problem | Trace order | Step types |
| --- | --- | --- |
| Expressions | Post-order evaluation of the tree: innermost first, left to right | `Add`, `Subtract`, `Multiply`, `Divide` |
| Equations | Expand products over `x` terms, combine like terms, move the `x` terms, move the constants, divide | `Expand`, `Combine`, `Subtract … from both sides`, `Add … to both sides`, `Divide` |

**Every step type that a held-out problem needs is used by training problems.** Phase A tests
this.

**Number ranges** (operand magnitude):

| Use | Range | Digits |
| --- | --- | ---: |
| In distribution (train, val, in-distribution test) | 0–999, mixed toward small values | 1–3 |
| Extrapolation (test) | 1,000–99,999 | 4–5 |

The extrapolation suite becomes a pure *digit-length* generalization test. That is known to be
hard for small transformers, and v3 does not assume it will be solved. It is a secondary suite.

**Verifier.** The final-answer verifier is unchanged. A new step checker re-derives every gold
trace step, including `Expand` and `Combine`, and runs on a sample of every split at build time.

### 3. Held-out *shapes* instead of held-out templates

**Shape.** A problem's shape is its tree topology plus the operator at each internal node, with
operands abstracted away. For equations, it is the pair of side shapes plus where `x` sits.

**Split by shape, seeded, before any rows are generated:**

| Shape set | Share of eligible shapes | Used for |
| --- | ---: | --- |
| train-visible | ≈ 80% | training, val, in-distribution test |
| **held-out-val** | ≈ 10% | development checks and the capability gate |
| **held-out-test** | ≈ 10% | the confirmatory H4 test only |

Eligible shapes are those with at least 4 operands or depth at least 3. Smaller shapes are all
train-visible, so the basic operations are always learnable.

**Coverage constraint, checked by the build.** Every operator, every parent–child operator pair
and every step type that appears in either held-out set also appears in train-visible shapes. A
held-out shape is then a **new combination of familiar parts**, not a new trace format. This is
the "systematic compositional generalization" sense of held-out.

**Development without touching the test.** Held-out-val and held-out-test have disjoint shapes.
Every development decision, including the capability gate, uses held-out-val, so the test is not
consulted.

**Audits** (the same build gates as v1):
- byte-identical regeneration;
- verifier pass on every record;
- semantic-fingerprint dedup across all splits;
- no shape overlap between the three shape sets;
- the general-corpus contamination scan.

### 4. Generation length

Traces and digit tokens are longer, so the generation caps rise from 128 to **320** new tokens
for evaluation and **256** for GRPO rollouts. SFT `sequence_length` is **384**. All three were set
from Phase A measurements; see the amendments at the end.

## Stages

These follow the spec stage graph. The stage recipes are protocol v2's unless stated.

| Stage | Change from protocol v2 |
| --- | --- |
| v0-base | Tokenizer v2 and stream v3; 1.0 B processed tokens, 70/30 general/math |
| v1-math | 200 M tokens, 85/15 math/general, curriculum easy → hard by depth |
| v2-sft | The same SFT recipe (75/25 show-work/direct), on stream-v3 problems from train-visible shapes |
| **Capability gate** | See below; decides whether Phase C runs |
| v2-extra-sft | Time-matched to v4-grpo on the same pod |
| v4-grpo | rl-v2's recipe at LR 1e-5 (KV cache, dynamic sampling, group 8), with **no LR search**. RL pool: train-visible shapes only, weighted toward deep shapes. |
| v3-dpo | **Dropped from v3** (recommended). H3 was already tested and not supported; this keeps cost and garden-of-paths down. |

**Capability gate (Q1), declared now.** Phase C runs only if v2-sft meets both conditions on
**held-out-val** (greedy show-work):
- accuracy is at least **10%**;
- sampled pass@4 is at least **20%**.

Otherwise v3 reports Q1 as "not achieved", stops, and the test stays sealed. The thresholds make
sure GRPO's prompts can produce mixed groups on structures related to held-out shapes, and that a
2-point H4 effect is not lost to floor effects.

**RL reward-variance gate.** The v1 rule (mixed groups ≥ 0.20, malformed ≤ 0.10, pass@1 between
0.05 and 0.95) is re-run on the v3 RL pool with v3 sampling settings.

**Seeds.** v4-grpo and v2-extra-sft each run with **2 training seeds**; the spec asks for this
before a broad claim. Pretraining, mid-training and SFT use 1 seed, as before.

## Primary analysis (to be frozen in `reports/confirmatory_plan_v2.md` before the test is opened)

| Item | Plan |
| --- | --- |
| Suite | held-out-test: new, frozen at build, sealed |
| Decoding | greedy show-work, 320 new tokens, eval protocol `eval-v2-cached` (the cached generator, for every branch) |
| Comparisons | v4-grpo − v2-sft and v4-grpo − v2-extra-sft, for each seed pair |
| Rule | spec §7: at least +2 points, paired 95% interval above 0 |
| Seeds | **H4 is supported only if the rule holds for both seeds** in both comparisons |
| Secondary | in-distribution test, extrapolation test, sampled pass@k, by-depth accuracy, failure-mode profile |

## Phases and costs

| Phase | Where | What | Estimated cost |
| --- | --- | --- | --- |
| A | Mac, CPU only, tiny models | Tokenizer v2 code; generator v3, shape split, coverage check and step checker; length measurements; tests; small local data samples (thousands of rows) | free |
| B | 1 pod session | Full data build (millions of rows, on the pod); tokenizer v2 training; v0-base → v1-math → v2-sft; val, in-distribution and held-out-val evals; capability gate | **≈ $3–4** (protocol v2's same stages cost $2.19 on an RTX 5090; digit tokens add length) |
| C | 1 pod session, only if the gate passes | RL gate; 2 × v4-grpo + 2 × v2-extra-sft; evals; confirmatory test | **≈ $2–3** (rl-v2 cost $2.31 with 2 GRPO runs and 1 control) |
| **Total** | | | **≈ $5–7**, unmeasured |

**Proposed caps,** to be confirmed by the owner: Phase B **$6 / 4 h**, and Phase C **$5 / 3.5 h**.
The usual safeguards apply: watchdog, termination on completion, and per-session approval.

## Risks, declared now

- **The capability gate may fail.** A 43 M-parameter model may not generalize compositionally
  even with tree data. That outcome is informative, and v3 stops cheaply if it happens.
- **Digit-length extrapolation** will likely stay near the floor. It is secondary for that reason.
- **Mixed shapes may make training harder.** In-distribution accuracy may drop below v2's 90%,
  because tree data is more varied than 14 templates. That is acceptable: v3 is measured on its
  own baselines, not against v2's numbers.
- **The coverage constraint may be too generous.** Held-out shapes that are new pairings of
  familiar parts could be easy. The capability gate and the per-depth breakdown will show this,
  and the confirmatory plan reports results by held-out depth.
- **Failure analysis was done on the v1 test suites.** That analysis motivated this design. No v1
  test content enters v3 data: the v3 generator and held-out sets are new, and the v3 test suites
  are built and frozen fresh.

## Phase A results and frozen settings (amendments, all before any v3 run)

Phase A built and tested the v3 code locally (CPU, tiny models, small samples). These settings
replace or sharpen the draft above.

**Generator** (`src/microreason/data/trees.py`):
- **Every sample verifies.** 2,976 of 4,000 in-distribution draws were accepted. All of them
  pass the independent SymPy check and the step checker. Rejections were for caps, division by
  zero, and no unique solution.
- **Mix:** expressions are about 64% of accepted problems and equations about 36%. About 57% of
  expression answers and 67% of in-distribution equation answers are integers.
- **Shape split** (fixed by a hash, pinned in tests):

  | | Train-visible | Held-out-val | Held-out-test | Eligible |
  | --- | ---: | ---: | ---: | ---: |
  | Expression shapes | 37,768 | 4,591 | 4,589 | 46,912 |
  | Equation shapes | 137 | 10 | 9 | 104 |

- **Coverage check:** passes. No operator pairing or step type in either held-out set is missing
  from training.
- **Equations:** the held-out equation sets are small (10 and 9 shapes), so most of the
  held-out signal comes from expressions. Results will be reported by kind.

**Tokenizer v2** (`configs/tokenizer_v2.yaml`): `Digits(individual_digits=True)`, then
byte-level.
- **Trade-off:** every digit is one token in every position. The cost is a separate space
  token before a number. A variant that attached the space to the first digit was rejected,
  because it gives leading digits two identities.
- **Training data balance:** the general-text sample is raised to 2,900 per 100 k, keeping
  v1's ~70/30 general/math character balance with the longer v3 problems.

**Lengths** (provisional digit tokenizer; `reports/v3_phase_a_lengths.json`): a v3 problem is
about **90 plain-text tokens** (p99 217), against 23 in v1. Show-work answers:

| Suite | Mean | Max |
| --- | ---: | ---: |
| In distribution | 75 | 220 |
| Held-out-val | 105 | 207 |
| Extrapolation | 95 (p99 249) | 261 |

These set the following limits:

| Setting | Value | Reason |
| --- | --- | --- |
| Evaluation `max_new_tokens` | **320** | Covers extrapolation's longest answer (261) |
| GRPO rollouts `max_new_tokens` | **256** | Rollouts use in-distribution prompts only |
| SFT `sequence_length` | **384** | Longest of 30,000 examples is 259; SFT never truncates |
| SFT batch | 64 examples per step | Unchanged from v2 |

**Evaluation protocol.** Every v3 checkpoint is evaluated with the **cached generator**
(`eval-v2-cached`, `--kv-cache`), with the same settings for every branch.
- **Why:** answers are about 3× longer, so uncached evaluation would cost about 25 minutes per
  sampled held-out pass.
- **Parity:** the cached generator matched the uncached one on 512/512 fp32 greedy prompts on
  the real v2-sft model (`reports/rl_v2_results/kv_check.json`), and in the local tests.
- **Comparability:** no v3 number is compared with a v1 or v2 number.

**Data sizes** (`configs/data_v3.yaml`): the stream has **6 M records**, with shard a at 60%.
- **Base pretraining:** shard a (about 324 M tokens) covers its 300 M math tokens in under one
  pass.
- **Mid-training:** shard b covers about 41 M easy tokens in about one pass and about 130 M hard
  tokens in under one pass.
- **Curriculum buckets:** easy is levels 1–2 (about 20% of tokens); hard is levels 3–5.
- **Other sets:** the SFT pool has 100,000 records; val and `test_id` have 7,000 each; each
  held-out suite has 6,000; extrapolation has 7,000.

**Stage configs** (`configs/pretrain_v3.yaml`, `midtrain_v3.yaml`, `sft_v3.yaml`): protocol v2
recipes with tokenizer v2, data v3, new seeds (52, 53, 54) and the SFT length above.

**Phase B session** (`scripts/pod_setup.sh` with `SETUP_PROFILE=v3`, then
`scripts/pod_v3_phase_b.sh`):
1. On the pod, build, audit and freeze the v3 suites. Build the SFT split, the stream and
   tokenizer v2, then tokenize.
2. Train v0-base, v1-math and v2-sft.
3. Evaluate on val: H1, H2, show-work with samples, and direct.
4. Evaluate on held-out-val: v2-sft show-work (greedy plus 4 samples) and direct, and v1-math
   plain.
5. Apply the capability gate (`scripts/v3_capability_gate.py`).

The test suites stay sealed: the script refuses any `test_*` evaluation. The frozen v3
manifests are pulled and committed after the session.

**Phase B estimate:**
- **Time:** about 2.5–3 h on one GPU. Setup and data take about 40 minutes, base pretraining
  about 60–80 minutes, and mid-training, SFT and evaluations the rest.
- **Cost:** **about $3–3.5** at the RTX 6000 Ada's $0.99/h.
- **Proposed bounds:** **$6 and 4.5 h** (`configs/budget_v3_phase_b.yaml`, which stays unpaid
  until approved).

## Phase C settings (declared 2026-11-09, after Phase B and before any Phase C run)

The capability gate passed (`reports/v3_phase_b.md`: v2-sft scored 70.3% greedy and 74.4%
pass@4 on held-out-val). The settings below are fixed for Phase C.

**RL pool** (`configs/rl_pool_v3.yaml`): a separate training-only seed stream (root seed
20261111) of 2 M records, excluding every frozen v3 evaluation record.
- **Family weights:** a record is kept with probability w_f / max(w), where
  w_f = 1 − p_f⁸ − (1 − p_f)⁸. Here p_f is v2-sft's greedy show-work **val** accuracy for its
  family.

  | Family | Keep probability |
  | --- | ---: |
  | 1 operator | 0.356 |
  | 2 operators | 0.846 |
  | 3 operators | 0.980 |
  | 4 operators | 1.000 |
  | 5 operators | 0.981 |
  | Linear equations | 0.293 |

- **Partitions** (`rl.pool`, by id hash):

  | Partition | Records | Use |
  | --- | ---: | --- |
  | gate | ~10 k | Reward-variance gate |
  | grpo | ~500 k | GRPO prompts |
  | dpo | ~290 k | Unused |
  | sft_extra | ~1.2 M | Extra-SFT examples |

**Checks before any GRPO update:**
- **KV-cache parity:** the check is re-run on p3-v2-sft with tokenizer v2, under the rl-v2 rule
  (fp32 greedy agreement at least 0.99; bf16 sampled pass@1 gap at most 0.02).
- **Reward-variance gate:** re-run on 512 gate prompts with GRPO's exact sampling (group 8,
  temperature 0.8, 256 tokens), under the v1 rule. The group size is now 8.

If either fails, the session stops and the test suites stay sealed.

**Training branches:**
- **v4-grpo-s1 and v4-grpo-s2** (`configs/grpo_v3_seed{1,2}.yaml`, seeds 61 and 62): rl-v2's
  selected recipe, with **no LR search**. LR is 1e-5, with dynamic sampling (32 mixed groups per
  step, chunks of 64, at most 8 rounds), group 8, 300 steps, 2 updates per step, clip 0.2, k3 KL
  β 0.04, and KV-cache rollouts of at most 256 tokens.
- **v2-extra-sft-s1 and v2-extra-sft-s2** (`configs/extra_sft_v3_seed{1,2}.yaml`, seeds 71 and
  72): the v3 SFT recipe at a constant LR of 5e-6, on 1 M `sft_extra` examples. Each is stopped
  at the matching GRPO seed's measured `train_wall_seconds` on the same pod.

**Evaluations:**
- **During training:** every branch, plus v2-sft re-evaluated on the same pod, is evaluated on
  val and held-out-val (greedy show-work).
- **Test suites:** opened once, afterwards, under `reports/confirmatory_plan_v2.md`.
  - **H4** needs all four GRPO comparisons, against v2-sft and against the matched control in
    each seed, to reach at least +2 points with intervals above 0.
  - **H1 and H2** get confirmatory checks on `test_id` (interval above 0).
  - v0-base and v1-math are uploaded to the pod for H1 and H2.

**Phase C estimate** (`scripts/pod_setup.sh` with `SETUP_PROFILE=v3c`, then
`scripts/pod_v3_phase_c.sh`):

| Step | Time |
| --- | --- |
| Setup and RL pool | about 15 min |
| KV check and gate | about 5 min |
| 2 × GRPO | about 20–25 min each |
| 2 × control | the same |
| Val evaluations | about 10 min |
| Test evaluations | about 40 min |

That totals **about 2.7–3 h**, so **≈ $3–3.5** on an RTX 6000 Ada at $0.99/h. The proposed
bounds are **$6 and 5 h** (`configs/budget_v3_phase_c.yaml`, which stays unpaid until approved).
GRPO step time is the main uncertainty, because v3 rollouts are about 2–3× longer than rl-v2's.
