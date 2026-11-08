# Stage 1 report: v0-base and v1-math (v1)

Date: 2026-11-08. One RunPod session: RTX 4090 Secure (US-NC-1), pod `asc1h83hr1eue8`, listed
$0.89/h. Owner-approved budget: $3 cap, 2 h limit (`configs/budget_stage1.yaml`).
Machine-readable results: `reports/stage1_v1/`.

**Scope and caveats:**
- **Val only.** All results are on the frozen **val** suite. Test suites stay sealed, so these
  numbers are development results, not the predeclared final H1 test.
- **One seed.** Each run used a single training seed.

## Spend and safety

| | Value |
| --- | --- |
| Pod lifetime | 42.9 min (04:42:01–05:24:51 UTC) |
| GPU cost at listed rate | ≈ $0.64 (storage excluded; the RunPod billing page is authoritative) |
| Budget guard | v0-base ledger $0.51 accrued at end, v1-math $0.57; no stops |
| Watchdog | armed before any work for a 06:42:01 UTC hard deadline; never fired |
| Termination | results and checkpoints copied and checksum-verified locally, then the pod was deleted; `get-pod` returns 404 and the pod list is empty |

Time split: setup and data rebuild ≈ 20 min, CUDA checks ≈ 1 min, v0-base ≈ 12 min, v1-math
≈ 4.5 min, both val evaluations ≈ 1.5 min, copy-back ≈ 2 min.

## Training

| Run | Tokens | Steps | Throughput (RTX 4090, bf16) | Peak GPU memory | Final val loss (general / math) |
| --- | ---: | ---: | ---: | ---: | --- |
| v0-base | 120 M (70% text / 30% `math_stream_a`) | 1,832 | ≈165 k tok/s (0.398 s/step) | 7.6 GiB | 3.2855 / 0.5685 |
| v1-math (curriculum) | 40 M (85% math from shard b / 15% text) | 1,221 | ≈159 k tok/s (0.206 s/step) | 7.6 GiB | 3.3164 / 0.5481 |

- **v0-base data:** no math problem was repeated (math stream shard a stayed in epoch 0).
- **v1-math curriculum:** phases changed at 10.0 M and 24.0 M tokens, as configured. Overall mix
  was 20.3% easy / 64.8% hard / 15.0% text. The easy bucket went into its second pass, as
  planned; the hard bucket stayed in its first.
- **Mid-training trade-off:** math val loss fell by 0.020 and general-text val loss rose by
  0.031, a small amount of forgetting despite the 15% replay.
- **4090 vs 5090:** the RTX 4090 runs at about 65% of the RTX 5090's measured 252 k tok/s.

## H1 on val (greedy, `plain-v1`, 7,000 prompts each)

| Checkpoint | Exact accuracy | Wilson 95% | Valid format | Truncated / verifier errors |
| --- | ---: | --- | ---: | --- |
| v0-base | 1,805 / 7,000 = **25.8%** | 24.8–26.8% | 100% | 0 / 0 |
| v1-math | 3,426 / 7,000 = **48.9%** | 47.8–50.1% | 100% | 0 / 0 |

**Paired difference (v1-math minus v0-base): +23.2 points; paired bootstrap 95% interval
[+22.1, +24.3]** (10,000 resamples over prompt ids, seed 0). On val, math mid-training clearly
improves exact-answer accuracy. The confirmatory H1 claim still needs the sealed test suites
under a predeclared plan, and ideally a second training seed.

### By template (val)

| Level | Template | v0-base | v1-math |
| --- | --- | ---: | ---: |
| 1 | add_ab | 17.8% | 41.4% |
| 1 | sub_ab | 19.6% | 48.2% |
| 1 | mul_ab | 38.8% | 79.2% |
| 2 | div_ab | 89.4% | 96.2% |
| 2 | ax_equals_b | 57.6% | 77.8% |
| 2 | x_plus_a_equals_b | 15.0% | 42.2% |
| 2 | add_mul_abc | 6.2% | 19.2% |
| 3 | ax_plus_b_equals_c | 38.0% | 66.6% |
| 3 | mul_sub_mul | 2.2% | 4.6% |
| 3 | sub_mul_add | 2.0% | 9.4% |
| 4 | ax_plus_b_equals_cx_plus_d | 32.2% | 69.0% |
| 4 | a_paren_x_plus_b_equals_c | 25.6% | 57.8% |
| 4 | a_times_paren_diff | 13.2% | 46.2% |
| 4 | paren_sum_times_diff | 3.4% | 27.4% |

### Reading the numbers carefully

- **`div_ab` mostly measures copying.** When `a/b` does not reduce, the answer is the prompt's
  own fraction. This inflates level 2 and explains why rational answers score higher than
  integers (v1-math: 71.5% vs 45.4%). Report it separately, or exclude it when summarizing
  arithmetic skill.
- **Signed addition and subtraction are weak.** They are the reason level 1 scores below level 2.
  `mul_ab` is comparatively strong at 79%.
- **Multi-step arithmetic is the weakest area.** `mul_sub_mul`, `sub_mul_add` and `add_mul_abc`
  stay below 20%. These need correct multi-digit products and correct combination. Equation
  templates improved the most (+29 to +37 points).
- **Completions are short and well formed.** Mean 30 tokens, max 56, so no completion came near
  the 128-token cap.

## Artifacts

- Final checkpoints (local, not tracked): `artifacts/v0-base` and `artifacts/v1-math` are
  symlinks into `runs/pod_results/asc1h83hr1eue8/`. Checksums were verified after the copy.
- Per-prompt rows: `runs/pod_results/asc1h83hr1eue8/runs/eval/*/rows.jsonl`.
- Summaries, H1 comparison, ledgers and CUDA checks: `reports/stage1_v1/`.

## Next

SFT (v2-sft) from v1-math, with the `sft-show-work-v1` and `sft-direct-v1` formats. Then the
equal-compute SFT control, DPO and GRPO branches (spec §6).
