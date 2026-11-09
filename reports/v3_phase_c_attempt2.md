# Protocol v3 Phase C, attempt 2 (temperature 1.0): stopped at the reward-variance gate

Date: 2026-11-09 UTC. This attempt ran under the recorded amendment in
`reports/protocol_v3.md`: GRPO sampling temperature raised from 0.8 to 1.0, gate rule unchanged.

**Session:** RunPod pod `w98fvcvwmviks0`, RTX 6000 Ada (Secure, US-WA-1), $0.99/h, from 05:07:47
to about 05:26, **≈ $0.30**. The pod is deleted. Results are in `reports/v3_phase_c_attempt2/`.

**Outcome:** the gate failed again. Per the amendment, the session stopped with no GRPO or
control run, **every v3 test suite stays sealed**, and no further sampling change is made under
protocol v3. **H4 is undecided under protocol v3.**

## Before the gate (all passed)

- **Data:** the regenerated suites matched every frozen hash, and the audit passed. The RL pool
  was byte-identical to attempt 1's (`0d333730…`).
- **Tokenizer and checkpoints:** tokenizer v2 verified; the three Phase B checkpoints uploaded.
- **KV-cache parity at temperature 1.0:**
  - fp32 greedy completions identical on 512/512 prompts.
  - bf16 sampled pass@1 0.651 for both samplers, a gap of 0.
  - Cached sampling was **9.4×** faster.

## The gate (512 gate prompts, 8 samples, temperature 1.0, 256 tokens)

| Measure | Attempt 1 (T 0.8) | **Attempt 2 (T 1.0)** | Rule |
| --- | ---: | ---: | --- |
| Mixed groups | 0.182 | **0.195** | at least 0.20: **failed** |
| All correct | 0.584 | 0.561 | — |
| All wrong | 0.234 | 0.244 | — |
| pass@1 | 0.667 | 0.656 | 0.05–0.95: passed |
| pass@8 | 0.766 | 0.756 | — |
| Malformed | 0.0002 | 0.0007 | at most 0.10: passed |

**Mixed fraction by family, attempt 2:**

| Family | Groups | Mixed | All correct | All wrong |
| --- | ---: | ---: | ---: | ---: |
| 1 operator | 19 | 0.00 | 1.00 | 0.00 |
| 2 operators | 129 | 0.21 | 0.64 | 0.16 |
| 3 operators | 120 | 0.22 | 0.56 | 0.23 |
| 4 operators | 98 | 0.19 | 0.42 | 0.39 |
| 5 operators | 73 | 0.22 | 0.25 | 0.53 |
| Equations | 73 | 0.16 | 0.82 | 0.01 |

## Reading

**Raising the temperature barely changed anything** (0.182 → 0.195).
- **Pattern:** v2-sft's outcomes are almost entirely decided by the problem, not by sampling
  noise. Each prompt is solved by nearly all 8 samples or by nearly none.
- **Coverage:** about 20% of prompts in every multi-operator family sit in between, and only those
  carry a GRPO learning signal.
- **Reaching the threshold:** it would take a larger change to sampling or to the training-only
  prompt distribution than this protocol allows.

**Lesson for any later protocol (post-hoc; not acted on here).** The 0.20 threshold was inherited
from v1, where every sampled group was trained on.
- **What rl-v2 and v3 actually do:** they use dynamic sampling, which trains only on mixed groups.
- **What 0.195 would cost:** about 164 sampled prompts, or 2.6 chunks of 64, per step to collect 32
  mixed groups. That is well within the 8-chunk limit.
- **The real question for dynamic sampling** is sampling cost per update, not a fixed fraction.

The threshold was not changed after seeing this result. Changing it would be moving the
goalposts.

## Status of protocol v3

- **Q1 (capability):** answered **yes** in Phase B (`reports/v3_phase_b.md`). With digit tokens
  and tree data, v2-sft solves 70.3% of held-out-shape val problems, the same as training shapes
  of equal size.
- **Q2 (H4):** **undecided.** Both pre-declared gate attempts failed narrowly, and the test suites
  were never opened.
- **H1 and H2:** supported on val in Phase B, with no test confirmation. Their confirmatory checks
  were part of the Phase C test phase, which did not run.

**Phase C spend:** two attempts, ≈ $0.33 + $0.30 = **≈ $0.63** of the $6 approved.
