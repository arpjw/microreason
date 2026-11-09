# Failure analysis v1: why every model fails out of distribution

Date: 2026-11-08. **Post-hoc and descriptive.** This analysis reads the greedy show-work rows
from the one confirmatory test evaluation (`reports/rl_v2.md`) for v2-sft, v4-grpo-r2 and
v2-extra-sft-r2. It does not revisit the H4 decision.
- **Reproduce:** `scripts/analyze_failures.py`. The output is in `reports/failure_analysis_v1.json`.
- **What the analysis compares:** each model trace is aligned with the gold `solution_steps`,
  and the first point of divergence is classified.
- **Test data stays out of training:** no test problem, trace or answer may enter training data.
  Because these findings come from test suites, any intervention they motivate must be judged on
  a **new** frozen held-out suite.

## Summary

The two out-of-distribution suites fail for **different, specific, mechanical reasons**. Neither
is a gap in arithmetic or algebra "reasoning" that RL on in-distribution prompts could fill.

1. **Numeric extrapolation is mostly a tokenizer and input-range problem.**
   - 90% of failures happen at step 1, where the model miscopies a prompt number. 82% of those
     miscopies keep only the number's first BPE token (`-330 + 167` → `-3 + 16`).
   - Every training operand is a single token. Extrapolation operands are two tokens 73% of the
     time.
   - When an extrapolation prompt's operands happen to be single tokens, copy errors fall from 92%
     to 26%, and the remaining failures are mostly arithmetic.
2. **Held-out templates fail by template substitution.**
   - In 99.97% of traces the model writes the step skeleton of a *training* template; it never
     composes a new one.
   - One of the three held-out templates needs step types (`Expand`, `Combine`) that appear in
     no training trace.
   - When the skeleton is borrowed, operands are bound by surface position as the borrowed
     template would bind them, so they are "copy" errors even though every number is a single
     token.
3. **In distribution, the failures are arithmetic.** Copy and skeleton errors there are below
   0.1%.

The three checkpoints have nearly identical failure profiles. Neither GRPO nor extra SFT changed
*how* the models fail out of distribution.

## First point of divergence (greedy show-work, share of all prompts)

| Suite | Model | Correct | Arithmetic | Copy (operands) | Skeleton | Wrong final only |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `test_id` | v2-sft | 90.2% | 9.7% | 0.03% | 0% | 0.1% |
| | v4-grpo-r2 | 87.8% | 12.1% | 0.03% | 0% | 0.07% |
| | v2-extra-sft-r2 | 94.3% | 5.7% | 0% | 0% | 0% |
| `test_extrapolation` | v2-sft | 0.2% | 9.7% | **89.7%** | 0.3% | 0% |
| | v4-grpo-r2 | 0.3% | 10.0% | **89.5%** | 0.3% | 0% |
| | v2-extra-sft-r2 | 0.2% | 9.4% | **89.9%** | 0.5% | 0% |
| `test_heldout_template` | v2-sft | 0.2% | 0.3% | 28.4% | **71.2%** | 0% |
| | v4-grpo-r2 | 0.07% | 1.6% | 19.6% | **78.8%** | 0% |
| | v2-extra-sft-r2 | 0.2% | 0.07% | 33.8% | **65.9%** | 0% |

**How each column is defined:**
- **Skeleton:** the operation word differs from gold, or the trace has a different number of
  steps.
- **Copy:** same operation as gold, but different operands.
- **Arithmetic:** same operation and operands as gold, but a wrong result.

The extra-SFT gain in distribution (+4.1 points on `test_id`) is entirely fewer arithmetic
errors (9.7% → 5.7%). GRPO's loss (−2.3) is entirely more arithmetic errors (9.7% → 12.1%).

## 1. Numeric extrapolation: copying multi-token numbers fails

**Where it fails:** step 1, in 6,973 of v2-sft's 6,985 failures. Of the 6,281 copy errors,
**5,133 (82%) are prefix truncations.** Every wrong operand is a leading-digit prefix of the
gold one. Examples: `167` → `16`, `-330` → `-3`.

**How the tokenizer splits operands** (byte-level BPE, 8,192 tokens, `artifacts/tokenizer/v1`):

| Operands in | Single token | Two tokens |
| --- | ---: | ---: |
| Training data (50 k-record sample) | 100% | 0% |
| `test_id`, `test_heldout_template` | 100% | 0% |
| `test_extrapolation` | 26.6% | **73.4%** |

- **Training operands:** in-distribution values are 0–99 (coefficients 2–12). Every one of them
  is a single token, such as ` 16` or ` 33`.
- **Three-digit numbers:** only 106 of the 900 values from 100 to 999 are single tokens.
  `167` becomes ` 16` + `7`, and `-330` becomes ` -` + `3` + `30`.
- **Outputs are different:** multi-token numbers are common as *results* in training. About 58%
  of training step results are 100 or more in magnitude, because products are long. The model
  writes long numbers but never has to read one from the prompt.

**The controlled comparison inside the suite** (v2-sft; the two other checkpoints are within
2 points):

| Extrapolation prompts | Prompts | Correct | Copy error | Arithmetic error |
| --- | ---: | ---: | ---: | ---: |
| Every operand is a single token | 203 | **5.4%** | 26.1% | 68.5% |
| Some operand is multi-token | 6,797 | 0.1% | **91.6%** | 8.0% |

**Interpretation:**
- **Copying:** with single-token operands, most copy errors disappear.
- **Arithmetic:** the model mostly cannot compute with out-of-range values even when it copies
  them correctly. Accuracy is only 5.4% on those 203 prompts.
- **Natural control:** `ax_equals_b` has coefficients of 13–40, still single tokens. Its failures
  are 97% arithmetic.
- **The cause:** extrapolation is blocked at the input by tokenization and training's narrow
  operand range, and then at the arithmetic.

## 2. Held-out templates: substituting a known template

**The model always reuses a training skeleton.** A skeleton is the sequence of operation words
in the trace. v2-sft's trace matched some training template's skeleton in **5,998 of 6,000**
held-out prompts:

| Reproduced training template | v2-sft | v4-grpo-r2 | v2-extra-sft-r2 |
| --- | ---: | ---: | ---: |
| `paren_sum_times_diff`: Add > Subtract > Multiply | 2,417 | 2,171 | 2,755 |
| `ax_plus_b_equals_cx_plus_d` (variants) | 1,902 | 1,942 | 1,722 |
| `a_times_paren_diff`: Subtract > Multiply | 804 | 1,040 | 511 |
| `mul_sub_mul`: Multiply > Multiply > Subtract | 772 | 759 | 729 |
| other training templates | 103 | 73 | 282 |
| **no training skeleton** | **2** | **15** | **1** |

**By held-out template** (v2-sft):

| Held-out template | Gold skeleton | Failure profile | What happens |
| --- | --- | --- | --- |
| `heldout_paren_quotient_plus`: (a+b)/(c−d)+e | Add > Subtract > Divide > Add | copy 1,134, skeleton 851 | Shares its first two steps with `paren_sum_times_diff`, which the model copies, often with the wrong operands. It then multiplies instead of dividing. |
| `heldout_nested_product`: a·(b+c·d)−e | Multiply > Add > Multiply > Subtract | skeleton 1,432, copy 567 | Treated as `mul_sub_mul` (a·b − c·d). On `-88·(93+(-36)·(-1))−(-68)` it computes `-88·(-1)` and `-68·(-68)`: it pairs numbers by position, not by parse. |
| `heldout_paren_both_sides`: a(x+b)+c = dx+e | **Expand** > **Combine** > Subtract > Subtract > Divide | skeleton 1,990, correct 10 | Gold needs `Expand` and `Combine` steps, which appear in **no training trace**; training uses only Add, Subtract, Multiply and Divide. The model imitates the both-sides equation template and produces nonsense steps. |

**Where the "copy" errors come from:** on held-out templates, 65% of v2-sft traces (3,880 of
6,000) use only numbers that appear in the prompt in their first step, combined in the wrong
pairs. The other 35% include at least one invented number. Every
number in these prompts is a single in-range token, so this is **operand binding by surface
pattern, not a tokenization failure**.

## What this means for the hypotheses and for a v3

**H4.** With in-distribution prompts and in-range numbers, GRPO's reward never comes from a
held-out composition or a multi-token operand. The policy has no sampled successes in those
regions to reinforce, because pass@4 there is 0.3%. GRPO can only sharpen what the model can
already sometimes do. This mechanism explains the floor result without any appeal to the RL
recipe.

**A held-out suite that cannot discriminate.** The held-out templates are built to need
structure no training template shows. That makes them a hard test of generalization by
composition, and the current models have none. One template also needs step *types* absent from
the training trace vocabulary, so its gold trace is unreachable by imitation.

**What a v3 would have to change** (proposals, not run):
- **Number representation.** Digit-level tokenization of numbers, or right-to-left digit
  grouping, so a 3-digit operand reads as the same digits a 2-digit one does. Train on operand
  ranges that include multi-token numbers, and keep a disjoint extrapolation range.
- **Compositional training data.** Generate random expression trees over the train-visible
  operations and depths, so a "new template" is a new combination of familiar parts rather than
  a new trace format. Give every step type a held-out problem needs at least one train-visible
  use.
- **A new frozen held-out suite,** declared before any v3 run, at a difficulty where the
  pre-RL model is measurably above zero. The current test suites have been used.

## Caveats

- The alignment heuristic compares step text with the gold trace. A model that solved a problem
  by a valid but different route would be labelled "skeleton" or "copy". On these suites, at
  most 0.2% of answers are correct by any route, so this does not change the profile.
- These are single-seed checkpoints. The failure profiles are near-identical across three
  differently trained checkpoints, which suggests they are stable.
