# Math training stream v1

Date: 2026-11-08. Fixes the math-data shortfall found in Phase 3. The planned processed math
tokens were about 70 M (36 M base + 34 M mid). The v1 train split held 2.26 M tokens, so that
plan would have repeated it about 31 times.

## What was built

- `microreason data generate-stream` (`configs/data_stream.yaml`) produces a separate
  `train_stream` split with the same generators and in-distribution ranges as v1.
- **Frozen evaluation problems are excluded.** The four frozen suite hashes are verified first,
  then every fingerprint and normalized prompt in them is blocked.
- **Small templates are used up completely.** Capacity-limited templates (single operations
  over |n| <= 99) contribute every unique problem not reserved by the frozen suites. The eight
  large templates get targets sized to roughly equal token shares.
- **Every record is verified.** Each one passed the independent SymPy check and the step check
  at build time.
- **The shards are disjoint.** A hash of each record's id assigns it to shard `a` (base
  pretraining) or shard `b` (mid-training), so neither stage repeats the other's problems.

## Result

| | Value |
| --- | --- |
| Records | 1,474,122 (built in about 4 minutes) |
| By difficulty | L1 76,400 · L2 279,722 · L3 466,000 · L4 652,000 |
| Capacity-limited (exhausted) | add_ab 18,900 · sub_ab 38,601 · mul_ab 18,899 · div_ab 38,402 · x_plus_a_equals_b 38,402 · ax_equals_b 6,918 |
| Shard a (base) | 758,140 problems, **36.09 M tokens** (needs 36.0 M) |
| Shard b (mid) | 715,982 problems, **34.10 M tokens** (needs 34.0 M) |
| Audit (`data audit-stream`) | 0 fingerprint or prompt overlaps with the frozen suites, 0 duplicates, 29,405 random records re-verified with 0 failures |

Counting the `[bos]`/`[eos]` separators added in packing, each shard covers its stage's planned
math tokens in at most one pass. Levels 1–2 are smaller than levels 3–4 because of the capacity
limits above, not by choice. Single-operation arithmetic still appears as the individual steps
inside every multi-step trace.

## Wiring

- `configs/pretrain.yaml`: 70% general text / 30% `math_stream_a_v1`. Checked locally: the
  mixture tracks 70.01% / 29.99%.
- `configs/midtrain.yaml` (new): 85% `math_stream_b_v1` / 15% general-text replay, 40 M tokens,
  1,221 steps. Its parent path is a placeholder until v0-base exists. The easy-to-hard
  curriculum sampler is **not implemented yet**; this config samples uniformly.
- `scripts/pod_setup.sh` rebuilds and audits the stream on a pod, and
  `scripts/expected_fingerprints.json` now covers both shards.

## Found while checking

The Mac cannot run full-size steps of the real pretraining config. bf16 autocast on MPS did not
finish one step in 10 minutes, and fp32 took 595 s for one 65,536-token step. Real stages run on
the GPU (≈252 k tok/s measured). The trainer now refuses bf16 on MPS instead of crawling.
