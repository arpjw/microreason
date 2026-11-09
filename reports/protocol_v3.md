# Protocol v3 (DRAFT for owner review): a capability-first redesign, then H4 again

Date: 2026-11-08. **Status: draft.** Nothing in this protocol has been run, and no money is
committed. Phase A (local code and tests) can start once the owner approves this design. Phases B
and C each need their own spend approval. Before the first pod run, the frozen version of this
protocol is committed with every open choice resolved.

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

Traces and digit tokens are longer, so the generation cap rises from 128 to **256** new tokens
for evaluation and rollouts. Phase A sets the SFT `sequence_length` from the measured maximum
example length; it stays at 256 if everything fits.

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
| Decoding | greedy show-work, 256 new tokens, eval protocol eval-v1 (the uncached generator) |
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

## Decisions needed from the owner

1. Approve this design so Phase A can start (free).
2. Number ranges: in distribution 0–999, extrapolation 4–5 digits. Or keep a narrower range,
   such as 0–99 in distribution, which is easier.
3. Drop DPO from v3 (recommended), or keep it.
4. Two seeds for the RL branches (recommended; about +$1), or one.
5. Phase B and Phase C spend caps. These will be asked for separately, after Phase A, with
   measured length and cost estimates.
