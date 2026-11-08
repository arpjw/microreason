# Protocol v2 results: scaled base, mid-training and SFT

Date: 2026-11-08. Protocol: `reports/protocol_v2.md` (recorded before any sealed test was opened).
One RunPod session: RTX 5090 Secure (EU-RO-1), pod `60s4fcs1xvk7mh`, listed $1.19/h, under the
owner's pre-approval. Bounds: $8 cap, 4.5 h (`configs/budget_scale_v2.yaml`).
Machine-readable results: `reports/scale_v2_results/`.

**Scope and caveats:**
- **Val only.** All accuracy numbers are on the frozen **val** suite (7,000 prompts, greedy).
  Test suites remain sealed.
- **One seed.**

## Spend and safety

| | Value |
| --- | --- |
| Pod lifetime | 110.4 min (06:57:33–08:48 UTC) |
| GPU cost at listed rate | ≈ $2.19 (storage excluded) |
| Budget ledgers at end of each run | v0-base $1.61, v1-math $1.87, v2-sft $1.91; no stops |
| Watchdog / grace | hard deadline 11:27:33 UTC armed first; a 45-minute self-delete was armed after the session; neither fired |
| Termination | 3 checkpoints copied and checksum-verified, `/root/collected` set, pod deleted (404, pod list empty) |

**Incident:** `pod_pull.sh` only read final-checkpoint lists from `reports/stage*/`, so the first pull
copied results but no checkpoints. Local verification caught it before the pod was released. The
checkpoints were copied directly and the script was fixed to include `reports/scale_*/`.

## Data and training

- **Stream v2:** 9,212,068 unique problems (148,055 cross-template duplicates dropped at merge).
  Audit passed: 0 overlaps with frozen suites, 0 duplicates, sampled re-verification clean.
  Shard a has 5.88 M problems / 297.6 M tokens (base). Shard b is split into easy 24.5 M and
  hard 144.4 M tokens (mid). The simple templates hit exactly the same capacity limits as in v1.

| Run | Tokens | Steps | Throughput (RTX 5090) | Final val loss (general / math) | Data passes |
| --- | ---: | ---: | ---: | --- | --- |
| v0-base | 1.0 B | 15,259 | ≈268 k tok/s (0.244 s/step), 7.6 GiB | 2.365 / 0.499 | math shard a < 1 pass; general text ≈ 1.8 passes (epoch 1) |
| v1-math | 200 M | 6,104 | ≈263 k tok/s | 2.426 / 0.542 | easy bucket ≈ 1.65 passes, hard < 1 |
| v2-sft | 100,032 examples | 1,563 | ≈129 k tok/s | show-work 0.0135 / direct 0.852 | 2 epochs of 50 k |

The curriculum switched phases at 50.0 M and 120.0 M tokens, as configured.

**Loss and accuracy disagreed in mid-training.** Math val loss *rose* during v2 mid-training
(0.499 → 0.542), yet exact accuracy rose 5.8 points. Val loss averages over every token, including
steps and wording; it is not a reliable stand-in for exact-answer correctness here.

## Accuracy on val (greedy)

| Checkpoint | Format | Protocol v1 | **Protocol v2** | Paired difference v2 − v1 (95% bootstrap) |
| --- | --- | ---: | ---: | --- |
| v0-base | plain-v1 | 25.8% | **88.1%** | **+62.3** [+61.1, +63.4] |
| v1-math | plain-v1 | 48.9% | **93.9%** | **+44.9** [+43.7, +46.1] |
| v2-sft | sft-show-work-v1 | 50.0% | **89.9%** | **+39.9** [+38.7, +41.0] |
| v2-sft | sft-direct-v1 | 21.8% | **38.2%** | **+16.4** [+15.4, +17.4] |

All v2 runs produced 100% valid answers. Sampled show-work for v2-sft (temperature 0.8, 4
samples): pass@1 86.2%, **pass@4 93.6%** (v1: 40.1% / 60.5%).

### Hypotheses at v2 (val, paired)

- **H1** (mid-training improves math): v1-math − v0-base = **+5.8 pts [+5.2, +6.5]**.
  Supported, as at v1 (+23.2), but the gain is much smaller once the base is well trained.
- **H2** (SFT gives instruction-format answers): show-work **+89.9 pts**, direct **+38.2 pts** over
  v1-math under the same SFT prompts. Supported; v1-math produces no answer blocks under SFT
  prompts.
- **SFT costs some accuracy.** v2-sft show-work (89.9%) is below v1-math's own plain format (93.9%).
  This is not a paired test, since the formats differ. The loss is concentrated in multi-step
  problems (see below). The same pattern appeared at v1.

### By template (val)

| Level | Template | v1-math (v1) | v0-base (v2) | v1-math (v2) | v2-sft show (v2) | v2-sft direct (v2) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | add_ab | 41.4% | 95.2% | 99.4% | 99.0% | 87.2% |
| 1 | sub_ab | 48.2% | 97.8% | 99.6% | 99.4% | 86.4% |
| 1 | mul_ab | 79.2% | 99.2% | 100.0% | 99.6% | 80.6% |
| 2 | div_ab | 96.2% | 92.8% | 97.0% | 98.2% | 97.0% |
| 2 | ax_equals_b | 77.8% | 81.8% | 93.2% | 98.4% | 83.0% |
| 2 | x_plus_a_equals_b | 42.2% | 90.6% | 97.4% | 96.0% | 82.6% |
| 2 | add_mul_abc | 19.2% | 93.6% | 97.8% | 91.8% | 0.8% |
| 3 | ax_plus_b_equals_c | 66.6% | 99.6% | 99.8% | 98.8% | 8.6% |
| 3 | mul_sub_mul | 4.6% | 42.0% | **57.8%** | 38.0% | 0.2% |
| 3 | sub_mul_add | 9.4% | 81.4% | 93.0% | 82.0% | 0.4% |
| 4 | ax_plus_b_equals_cx_plus_d | 69.0% | 96.6% | 99.8% | 98.8% | 4.6% |
| 4 | a_paren_x_plus_b_equals_c | 57.8% | 79.8% | 87.8% | 81.6% | 1.6% |
| 4 | a_times_paren_diff | 46.2% | 95.2% | 98.6% | 93.4% | 1.4% |
| 4 | paren_sum_times_diff | 27.4% | 87.2% | 92.8% | 83.2% | 0.2% |

- **Scaling fixed most of the v1 weaknesses.** Signed addition and subtraction went from about
  45% to 99%, and `a*(b-c)` and `(a+b)*(c-d)` from below 50% to above 92%.
- **`mul_sub_mul` (a·b − c·d) is the clear remaining weak spot:** 57.8% at best. It needs two
  exact two-digit products, which often reach 4 digits, so the tokenizer splits them into
  irregular pieces.
- **SFT's cost is concentrated** in `mul_sub_mul` (−20), `sub_mul_add` (−11), the parenthesized
  templates (−6 to −10) and `add_mul_abc` (−6). Single-step problems are unchanged or better.
- **Direct answers stay near zero on anything multi-step.** The model still relies on its trace.

## Where this leaves RL (for discussion)

- **The starting point is stronger.** GRPO would now start from v2-sft (89.9% greedy show-work)
  rather than 50%.
- **The headroom has shrunk.** Sampled pass@1 is 86.2% and pass@4 93.6%, so many RL-pool groups
  would be all-correct and carry no signal. The spec's reward-variance gate (§6, Stage 5) has to be
  measured on the training-only RL pool before spending on GRPO, and the pool may need to be
  weighted toward the hard templates.
- **The equal-compute SFT control** (v2-extra-sft) is required for H4, and is cheap.

## Artifacts

- Checkpoints (local, not tracked): `artifacts/p2-v0-base`, `artifacts/p2-v1-math`,
  `artifacts/p2-v2-sft` are symlinks into `runs/pod_results/60s4fcs1xvk7mh/`, checksum-verified.
- Per-prompt rows: `runs/pod_results/60s4fcs1xvk7mh/runs/p2/eval/*/rows.jsonl`.
- Summaries, comparisons (including `scaling_v1_vs_v2_val.json`), ledgers and the stream-v2
  manifest: `reports/scale_v2_results/`.
