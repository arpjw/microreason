# Protocol v4 Phase B: converged parent, gates, LR pilots

Date: 2026-11-09 UTC. Protocol: `reports/protocol_v4.md`, with amendment 1 (40-step pilots and
150-step Phase C runs), recorded before any pilot result.

**Session:** RunPod pod `8tzge9u80u3s4o`, RTX 6000 Ada (Secure, US-WA-1), $0.99/h.
- **Time and cost:** 07:01:16 → 10:13 UTC, about 3 h 12 min, **≈ $3.17** (cap $5 / 4.5 h).
- **Phase C** runs next on the same pod, under its own ledger.
- **Results:** in `reports/v4_phase_b/`.

**Scope:** held-out-VAL only. Every test suite stayed sealed.

## Data

| Pool | Records | sha256 | Partitions used |
| --- | ---: | --- | --- |
| SFT pool v4 | 5,000,000 | `76acc7c8…` | `sft_extra` 3,000,166 (v2-sft-long trains on the first 3 M) |
| RL pool v4 | 4,000,000 | `25a81015…` | `gate` 20,033; `grpo` 1,000,221; `sft_extra` 2,400,313 |

The v3 suites were regenerated and match their frozen hashes, and the audit passed. Both v4 pool
manifests are now tracked in `data/manifests/`.

## v2-sft-long: the converged-parent attempt

p3-v2-sft plus one epoch over 3 M fresh examples: 46,875 steps, 72.5 min of training. Startup
took about 13 min, spent building examples single-threaded; that is fixed for Phase C in commit
`93f8828`.

| Examples | Held-out-val greedy (show-work) |
| ---: | ---: |
| 100 k (p3-v2-sft, protocol v3) | 70.3% |
| 375 k | 62.3% |
| 750 k | 68.4% |
| 1.5 M | 74.6% |
| **3 M (v2-sft-long)** | **80.7%** |

**Not plateaued.** The last doubling added +6.1 points, against the declared 1.0-point
threshold. As declared, this is descriptive and the 3 M checkpoint is the parent. It means
continued SFT is still a strong competitor.

The 375 k dip comes from re-warming the LR to 5e-5; the cosine schedule recovers it.

| v2-sft-long | Val (in-distribution) | Held-out-val |
| --- | ---: | ---: |
| Show-work greedy | **86.7%** (1op 97.9, 2op 88.9, 3op 77.6, 4op 66.3, 5op 54.0, eq 99.3) | **80.7%** (3op 78.1, 4op 65.6, 5op 55.6, eq 99.0) |
| Direct greedy | 34.8% | — |

## Gates (all passed)

| Gate | Result | Rule |
| --- | --- | --- |
| Headroom (held-out-val, 16 samples, T 1.0) | pass@16 88.9% − pass@1 77.6% = **0.113** | ≥ 0.05 |
| KV-cache parity | fp32 greedy 512/512 identical; pass@1 gap 0; **23.3×** faster | as v3 |
| Filter yield (512 gate prompts, G 16, filter 2..14) | **0.213** (mixed 0.279, all-correct 0.570, all-wrong 0.150; pass@1 0.713) | ≥ 0.125 |

## LR pilots (held-out-val; selection rule fixed in advance)

**GRPO**, 40 steps × 1,024 trained completions, pilot seed 80:

| LR | Held-out-val | Δ vs parent | Training pass rate, first → last 10 steps | KL at end | Filter yield | Wall |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1e-6 | 80.87% | +0.17 | 0.716 → 0.706 | 0.0003 | 0.194 | 678 s |
| **3e-6 (selected)** | **81.07%** | **+0.37** | 0.717 → 0.716 | 0.0020 | 0.184 | 684 s |
| 1e-5 | 80.67% | −0.03 | 0.719 → 0.718 | 0.0050 | 0.171 | 723 s |

**Control** (continued SFT on RL pool v4 `sft_extra`), each time-matched to the selected GRPO
pilot (684 s):

| LR | Held-out-val | Δ vs parent | Steps / examples |
| --- | ---: | ---: | --- |
| **5e-6 (selected)** | **81.47%** | **+0.77** | 8,296 / 531 k |
| 2e-5 | 79.00% | −1.70 | 8,322 / 533 k |

**Step time** (the reason for amendment 1): a median of 16.7 s, of which about 85% is
generation, at about 3 sampling chunks per step.

## Reading (val only; no test data)

- **Training reward still does not rise within 40 steps**, even with every v4 change in place:
  on-policy updates, Dr. GRPO normalisation, 1,024 filtered samples per update, and published
  LRs. The bug check rules out a sign error.
  - **What does change:** the policy sharpens. Sampled-token NLL falls (0.019 → 0.016 at 1e-5)
    and filter yield falls with LR, so groups become more all-or-nothing.
  - **Interpretation:** this matches the literature's "RL squeezes" picture.
- **At equal time, continued SFT gained about twice what GRPO did** on held-out-val (+0.77 vs
  +0.37). The parent has not plateaued, and the control processes about 531 k fresh gold traces
  in the time GRPO trains on 41 k self-samples.
  - **For H4:** the Phase C comparison with the control looks unfavourable to GRPO.
  - **No change to the plan:** Phase C runs as pre-registered, because Phase B's job was only to
    select LRs.
- **The protocol's pilot rule was met:** GRPO's best pilot beat the parent (+0.37), so Phase C
  proceeds under the owner's pre-approval.
