# Stage 2 report: v2-sft (v1)

Date: 2026-11-08. One RunPod session: RTX 4090 Secure (EUR-IS-2), pod `0qmy7214qq4fmu`, listed
$0.89/h. Owner-approved budget: $2 cap, 1.5 h limit (`configs/budget_stage2.yaml`).
Machine-readable results: `reports/stage2_v1/`.

**Scope and caveats:**
- **Val only.** Results are on the frozen **val** suite; test suites stay sealed.
- **One seed.**

## Spend and safety

| | Value |
| --- | --- |
| Pod lifetime | 29.9 min (06:14:14–06:44 UTC) |
| GPU cost at listed rate | ≈ $0.44 (storage excluded) |
| Budget guard | $0.19 accrued at the end of the SFT run; no stops |
| Watchdog | armed before any work for a 07:44:14 UTC deadline; never fired |
| Termination | results and the v2-sft checkpoint copied and checksum-verified, then the pod was deleted (404, pod list empty) |

**Incident: macOS metadata in the uploaded parent checkpoint.** macOS `tar` added `._*`
AppleDouble files to the uploaded v1-math checkpoint, so checkpoint verification refused it
before any training started. The real files matched their sha256. The extra files were deleted,
the checkpoint re-verified, and the session restarted. Total delay was about 4 minutes. Future
uploads set `COPYFILE_DISABLE=1` (see the runbook).

## SFT run

| | Value |
| --- | --- |
| Parent | v1-math (`step_0001221`) |
| Data | 50,000 verified train records: 37,377 show-work / 12,623 direct (fixed hash rule); mean 45.7 tokens, max 86 |
| Schedule | 2 epochs = 100,032 examples, 1,563 steps of 64, peak LR 5e-5 cosine |
| Speed | 0.077 s/step, ≈38 k tok/s (short padded examples), peak 2.9 GiB |
| Final val loss on assistant tokens | show-work 0.077, direct 0.912 |

## H2 on val (identical SFT prompts, greedy, 7,000 prompts each)

| Checkpoint | Show-work accuracy | Direct accuracy | Valid format |
| --- | ---: | ---: | --- |
| v1-math | 0 / 7,000 = 0.0% | 0 / 7,000 = 0.0% | 0%: 81% / 96% truncated, rest invalid |
| v2-sft | 3,499 / 7,000 = **50.0%** (48.8–51.2%) | 1,527 / 7,000 = **21.8%** (20.9–22.8%) | 100% |

Paired differences (v2-sft minus v1-math): **show-work +50.0 pts [+48.8, +51.1]**, **direct +21.8
pts [+20.9, +22.8]** (paired bootstrap, 10,000 resamples). **H2 (format and instruction
following) is supported on val.** v1-math never produces an answer block under the SFT prompt.

Sampled show-work (temperature 0.8, 4 samples per prompt, seed 0): pass@1 40.1%, **pass@4 60.5%**.

## Did SFT raise accuracy?

Barely. v2-sft show-work scores 50.0% against v1-math's 48.9% in its own plain format. The
formats differ, so this is not a paired test.

| Level | Template | v1-math (plain) | v2-sft show-work | v2-sft direct |
| --- | --- | ---: | ---: | ---: |
| 1 | add_ab | 41.4% | 53.6% | 26.8% |
| 1 | sub_ab | 48.2% | 51.2% | 30.4% |
| 1 | mul_ab | 79.2% | 79.0% | 35.4% |
| 2 | x_plus_a_equals_b | 42.2% | 54.8% | 27.0% |
| 2 | ax_equals_b | 77.8% | 85.6% | 73.4% |
| 2 | div_ab | 96.2% | 98.0% | 93.8% |
| 2 | add_mul_abc | 19.2% | 19.8% | 2.6% |
| 3 | ax_plus_b_equals_c | 66.6% | 67.4% | 8.4% |
| 3 | mul_sub_mul | 4.6% | 4.2% | 0.6% |
| 3 | sub_mul_add | 9.4% | 7.2% | 0.8% |
| 4 | ax_plus_b_equals_cx_plus_d | 69.0% | 66.4% | 3.6% |
| 4 | a_paren_x_plus_b_equals_c | 57.8% | 51.8% | 2.6% |
| 4 | a_times_paren_diff | 46.2% | 39.0% | 0.0% |
| 4 | paren_sum_times_diff | 27.4% | 21.8% | 0.0% |

- **Single steps improved, longer traces got slightly worse.** Show-work SFT helps single-step
  problems (signed addition +12, one-step equations +8 to +13) and slightly hurts level-4
  parenthesized problems (−5 to −7). Possible causes, not yet isolated: drift from 2 epochs over
  a 50k set, or the 25% direct examples. The equal-compute SFT control (v2-extra-sft) and an
  ablation are needed before attributing it.
- **Direct answers need the steps.** Direct answers collapse on multi-step problems (0–9%), so
  the model relies on its written trace.
- **Multi-digit products remain the bottleneck.** `mul_sub_mul` and `sub_mul_add` are below 10%
  in every format.
- **There is room for RL.** pass@4 (60.5%) is far above sampled pass@1 (40.1%), so a large share
  of prompts have mixed right/wrong samples. That variance is the signal GRPO's exact-answer
  reward needs (spec §6, Stage 5).

## Artifacts

- Final checkpoint (local, not tracked): `artifacts/v2-sft` is a symlink into
  `runs/pod_results/0qmy7214qq4fmu/`, checksum-verified after the copy.
- Per-prompt rows: `runs/pod_results/0qmy7214qq4fmu/runs/eval/*/rows.jsonl`.
- Summaries, H2 comparisons, ledger and CUDA checks: `reports/stage2_v1/`.
