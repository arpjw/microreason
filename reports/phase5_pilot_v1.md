# Phase 5 report: GPU pilot (v1)

Date: 2026-11-08. Provider: RunPod Secure Cloud. Owner-approved: $5 cap, 3 h limit, terminate at
end. Machine-readable results: `reports/pilot_v1/`.

## Spend

| Pod | GPU | Region | Rate | Lifetime | GPU cost (listed rate) | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `5gqf57qzyj9ck6` | RTX 4090 | not reported | $0.89/h | ~3.4 min | ~$0.05 | Deleted before any work: self-termination could not be verified |
| `n4jju3kt4fza5f` | RTX 5090 | EUR-NO-1 | $1.19/h | ~49.5 min | ~$0.98 | Pilot completed, results copied, pod terminated |
| **Total** | | | | | **~$1.03** | Under the $5 cap. Storage is excluded; the RunPod billing page is authoritative |

Both pods were confirmed deleted (`get-pod` returns 404; the pod list is empty).

## Results (RTX 5090, 32 GB, bf16 autocast, torch 2.15.0+cu132)

| Check | Result |
| --- | --- |
| Test suite on CUDA | 251 collected: 250 passed, 1 skipped (MPS-only test) |
| Data rebuilt on pod from pinned sources | Fingerprints match the reference machine byte-for-byte (tokenizer, text split, math splits, all tokenized datasets) |
| Smoke run (3.3 M model, 300 steps) | ~230–260 k tok/s; final val loss 6.1386 general / 1.4969 math (Mac MPS fp32: 6.1422 / 1.5054) |
| Resume on CUDA | Resumed run reproduces step losses and final val loss exactly (6.1386 / 1.4969) |
| Budget guard | Ledger kept for every paid run; no stop needed; elapsed 0.81 h, accrued ≈ $0.96 at the end of the main run |

### Full 43 M model, sequence 1,024, 65,536 tokens per optimizer step

| Micro-batch × accumulation | Median step | Tokens/s | Peak GPU memory |
| --- | ---: | ---: | ---: |
| **16 × 4** | 0.257 s | **≈255 k** | 7.6 GiB |
| 32 × 2 | 0.276 s | ≈237 k | 14.4 GiB |
| 64 × 1 | 0.307 s | ≈213 k | 27.9 GiB |

200-step pilot run (micro-batch 16, 13.1 M tokens): median 0.260 s/step, **≈252 k tok/s**, peak
7.6 GiB. Train loss 4.11 at step 200, val loss 5.40 general / 1.34 math. Those losses are an
engineering signal only, not evidence of any capability.

## What this means

1. **Compute is not the constraint.** At the measured throughput, the planned stages need:

   | Stage | Tokens | GPU time | GPU cost at $1.19/h |
   | --- | ---: | ---: | ---: |
   | Base pretraining (v0-base) | 120 M | ≈8 min | ≈$0.16 |
   | Math mid-training (v1-math) | 40 M | ≈2.6 min | ≈$0.05 |

   These exclude evaluation, checkpointing and setup. Setup alone took ≈11 min (environment and
   data rebuild), so it costs more than the training itself. The proposed $50 project cap is far
   more than base and mid-training need. GRPO rollouts and evaluation generation will be the
   larger costs; they are unmeasured.
2. **Smaller micro-batches are faster here.** 16 × 4 beat 64 × 1 by about 20%, probably because
   the large fp32 logits tensor (sequences × 1,024 × 8,192) dominates memory traffic. Use
   micro-batch 16.
3. **The Phase 3 memory estimate was about 1.8× too low.** It estimated 15.1 GiB for micro-batch
   64 and the measurement is 27.9 GiB. That would not fit a 24 GB RTX 4090; micro-batch 16
   (7.6 GiB) or 32 (14.4 GiB) would.
4. **CUDA and the Mac agree closely but not exactly**, as expected across hardware and precision.
   Resume is exact within a device.

## Incidents and fixes

| Incident | Cost | Fix |
| --- | --- | --- |
| No RTX 4090 capacity in EU-RO-1, then US-IL-1, then anywhere | none (failed creates are free) | Owner approved any region, then an RTX 5090 at $1.19/h |
| SSH sessions do not inherit container env; the image ships runpodctl 1.14; the pod's built-in key got 403 even for a self-read | ~$0.05 (pod deleted per runbook) | `pod_env.sh` loads the container env; checksum-verified runpodctl 2.15; owner-provided watchdog key via the RunPod Secret `claude` |
| Container sees 120 host CPUs against a 12.75-CPU quota; pytest spawned 189 threads and stalled for ~31 min | ~$0.65 of idle GPU time | Thread pools capped at the cgroup quota; tests time-limited to 15 min |
| Test capture kept only the last 3 progress lines (no summary) | none | Full pytest output saved; summary line kept |

## Safety record

- **Watchdog:** armed before any work on pod 2, with a hard deadline of 04:11:26 UTC (creation + 3 h).
- **Approvals:** every change of region and GPU was approved in chat before use. The listed price
  was re-checked immediately before each create.
- **Results and termination:** results were copied off before termination, and both
  terminations were verified.
- **Credentials:** no credential values were printed, logged or committed. The watchdog key
  reached the pod only through the RunPod Secret reference.

## Next

Phase 5 is complete. Before Phase 6, the remaining blocker is data, not compute. The math
training split holds 2.26 M tokens against a planned ~70 M (see Phase 3 and 4 reports).
