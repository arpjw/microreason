# MicroReason closeout: what four protocols taught us about a 43M-parameter reasoner

Date: 2026-11-09. This report closes the project. It supersedes `reports/closeout_v1.md`, which
covered protocols v1 and v2 only.

## The question

Can a ~43M-parameter decoder, trained from random initialization through a staged pipeline,
learn arithmetic and basic algebra within a small compute budget? The pipeline is base
pretraining → math mid-training → SFT → RL. Four hypotheses were tested, each with a
pre-registered rule and sealed test suites:

| ID | Hypothesis | Final finding | Strongest evidence |
| --- | --- | --- | --- |
| H1 | Math mid-training improves math capability | **Supported** (confirmed on test) | v4 Phase C, `test_id`: v1-math − v0-base = +5.29 [+4.49, +6.14] |
| H2 | SFT gives instruction following and answer format | **Supported** (confirmed on test) | v4 Phase C, `test_id`: v1-math under SFT prompts 0.0%, v2-sft 79.5% (+79.5 [+76.6, +82.2]) |
| H3 | DPO improves selection or format without a major correctness loss | **Not supported** | v1 RL stage: format was already 100% valid; correctness −5.1 on `test_id` |
| H4 | Exact-outcome GRPO improves held-out correctness beyond SFT and equal-cost continued SFT | **Not supported** | v4 Phase C, `test_heldout_shape`: +0.49 [+0.09, +0.98] vs parent; −1.06 [−1.76, −0.53] vs time-matched continued SFT |

**One confirmed secondary finding.** RL preserves out-of-range ability that extra SFT erodes. On
4–5 digit extrapolation, GRPO beat time-matched continued SFT by +2.18 [+0.56, +3.88] in v4.
It did so again by **+2.17 [+0.62, +3.78] on a fresh, separately frozen suite**
(`reports/replication_v4x.md`), with all three seeds positive both times.

## How we got here

| Protocol | What changed | What it found | Spend |
| --- | --- | --- | ---: |
| **v1** (`reports/stage1_v1.md`, `stage2_v1.md`) | 14 fixed problem templates, BPE tokenizer, small budgets | The pipeline works: mid-training +23 pts on val; SFT 100% valid format | ≈ $1.1 |
| **v2** (`scale_v2.md`, `rl_v1.md`, `rl_v2.md`) | 1 B base tokens + 200 M mid tokens; GRPO, DPO and equal-compute SFT | val 88–94%, but **every model scored ≤ 0.2% on held-out templates**. GRPO and DPO lost to their parent; extra SFT won. H4 not supported, at the floor | ≈ $7.3 |
| **v3** (`protocol_v3.md`, `v3_phase_b.md`) | One token per digit; problems generated from random expression trees; held-out *shapes* | **Generalization fixed:** 70.3% on held-out shapes, as good as on training shapes of the same size. The RL gate failed twice (0.182 and 0.195 < 0.20), so no GRPO ran | ≈ $3.4 |
| **v4** (`protocol_v4.md`, `v4_results.md`) | A literature review; a parent trained much longer (80.7%); a pass-rate prompt filter; published-default on-policy GRPO; 3 seeds; RFT and step-reward arms | **GRPO helps for the first time** (+0.5), but **equal-time continued SFT helps more** (+1.5). H4 not supported. RL preserves extrapolation (exploratory) | ≈ $10.5 |
| **v4x** (`replication_v4x.md`) | Evaluation only, on a fresh frozen extrapolation suite | The extrapolation finding **replicated** | ≈ $1.0 |

**Total GPU spend: about $23**, almost all on RunPod RTX 4090, 5090 and 6000 Ada pods. Every pod
ran with a self-deleting watchdog and was deleted at the end.

## What we learned

1. **The data representation was the bottleneck, not the training method.**
   - **v1/v2's failure:** they failed out of distribution because multi-digit numbers were opaque
     BPE tokens, and a fixed template set taught 14 scripts instead of an algorithm
     (`reports/failure_analysis_v1.md`).
   - **The fix:** one token per digit, plus tree-generated problems with held-out shapes, took
     held-out accuracy from 0.2% to 70% with the same model and budget.
   - **What remains:** accuracy is now limited by depth (about 95% at 1 operator, 55–60% at 5),
     not by novelty.
2. **GRPO needs the published recipe even at 43M.**
   - **v1–v3 runs:** they used LR 1e-5, std-normalised advantages, KL 0.04, 256 samples per step
     and off-policy minibatches. They lost to their parent, and training reward never rose.
   - **v4's recipe:** on-policy updates, group-mean/batch-std advantages, a constant-length loss,
     LR 3e-6, a filtered stream at 1,024 samples per update, and KL 1e-3.
   - **Result:** small but consistent gains on every seed and every in-distribution suite. A test
     with a learnable stub reward ruled out a code bug (`tests/test_rl_v4.py`).
3. **With unlimited correct worked solutions, extra SFT beats RL at equal compute.**
   - **Throughput:** in GRPO's 46 minutes, continued SFT processed about 2.2 M fresh gold traces;
     GRPO trained on about 154 k of its own samples.
   - **The parent kept improving:** 70 → 81% on held-out-val as SFT grew from 0.1 M to 3 M
     examples, and it had not plateaued.
   - **Consistency with the literature:** this matches "RL sharpens, SFT expands" when a perfect
     teacher is free.
   - **Where H4 could be tested fairly:** only where gold traces are scarce or the SFT parent has
     truly plateaued.
4. **RL's advantage here is protection, not improvement.**
   - **Extra SFT erodes extrapolation:** continued SFT on in-range numbers cost 2.3–2.4 points on
     out-of-range numbers, twice, on independent suites.
   - **RL does not:** GRPO and RFT left it unchanged.
   - **Caveat:** absolute extrapolation accuracy is low (7–10%), so this is a robustness result,
     not a capability one.
5. **GRPO's negative term added nothing over RFT here** (+0.13, interval includes 0).
   - **What drives the gain:** training on filtered on-policy samples, as Xiong et al. found at
     scale.
   - **Process reward:** a bounded, ungameable step-prefix reward did not help either.
6. **Process lessons.**
   - **Discipline paid off:** pre-registration, sealed suites and amendments recorded before
     results kept every claim honest. Two RL gate failures and two negative H4 results were
     reported as they were, never reframed.
   - **Operational lessons:** `memory/microreason-pod-lessons`, `scripts/PILOT_RUNBOOK.md`,
     and the protocol amendments. Examples: never edit a running bash script on a pod; budget
     generation time realistically.

## State of the repository

- **Code:**
  - `src/microreason/`: model, tokenizer, data generators v1/v3, training stages, GRPO/DPO/RFT
    (`rl/`), evaluation with paired and cluster bootstrap.
  - About 350 tests.
- **Protocols and plans:**
  - `reports/protocol_v{2,3,4}.md`;
  - `reports/confirmatory_plan_v{1,2,3}.md`;
  - `reports/replication_v4x.md`;
  - literature review: `reports/GRPO gains for tiny models.md`.
- **Frozen data:**
  - manifests and frozen hashes for data v1, v3 and v4x (`data/manifests/`);
  - generated files are rebuilt deterministically from seeds.
- **Checkpoints** (local, not tracked): `runs/pod_results/<pod>/`.
  - `8tzge9u80u3s4o` holds v2-sft-long and all v4 branches.
  - `pppweshg0o8h9t` holds the v3 Phase B stages.
- **Test suites:**
  - Used: data v1 (v2 RL stage), data v3 (v4 Phase C), data v4x (replication). Any further
    evaluation on them is post hoc.
  - A future protocol needs freshly generated, separately frozen suites, which
    `generate-replication-v4x` shows how to make.

## If the project resumes

Most informative next experiment: **a scarce-gold-trace v5.**
- **Design:** cap SFT gold traces (for example 50k–200k) for both the parent and the
  continued-SFT control. GRPO gets unlimited problems with answer-only checking. Use fresh sealed
  suites.
- **Why:** this is the regime where outcome RL is supposed to beat SFT, and nothing tested so far
  has covered it.
- **Readiness:** the v4 code supports it with configuration changes only.
