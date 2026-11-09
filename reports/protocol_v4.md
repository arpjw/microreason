# Protocol v4: a converged parent, an informative prompt stream, and a published-default RL recipe

Date: 2026-11-08 (local).

**Status:**
- **Phase A done:** code, configs and tests, built locally on CPU (`tests/test_rl_v4.py`,
  `tests/test_v4_decide.py`).
- **Nothing has run on a pod.**
- **Approvals needed:** this design needs the owner's approval. Phases B and C each need their
  own spend approval: Phase B is about $3 (cap $5, 4.5 h), Phase C provisionally about $6.5
  (cap $9, 8 h), re-estimated after Phase B.
- **Sealed:** the data-v3 test suites (`data/manifests/math_eval_frozen_v3.json`) are unused, and
  this protocol opens them once, at the end of Phase C.

**Background:** the design follows the literature review in
`reports/GRPO gains for tiny models.md`, with notes in `research_notes/GRPO gains for tiny models/`.

## Why v4

Protocol v3 answered Q1: with digit tokens and tree data, v2-sft solves 70.3% of held-out-shape
val problems. It left H4 undecided, because both pre-declared reward-variance gate attempts
failed narrowly (0.182 and 0.195 < 0.20). Across v1 to v3, every GRPO run that did happen failed
the same way: training reward stayed flat, GRPO lost to its parent, and it lost by more to
time-matched continued SFT (rl-v2: −1.8 and −6.3 points on val).

The review points to five causes. Each one changes in v4:

| # | Cause in v1–v3 | v4 change | Evidence |
| --- | --- | --- | --- |
| 1 | **Undertrained parent.** v3's v2-sft saw 100 k examples in 2.3 min, so continued SFT on fresh gold traces was still in its steep phase. RL, which mainly sharpens, was racing an expansion method that had not converged. | Train **v2-sft-long** (+3 M fresh examples) before branching; measure its plateau. | Yue et al. 2025; "RL squeezes, SFT expands"; 45M from-scratch study (2510.15990) |
| 2 | **Uninformative prompts.** About 80% of groups were all-correct or all-wrong, and the pool was weighted only by family. | A **streaming pass-rate filter** keeps groups with 2–14 of 16 correct, on a **depth-weighted** stream (~70% at 4–6 leaves). | Online Difficulty Filtering (2504.03380); POLARIS; Skywork-OR1; DAPO |
| 3 | **Step size and regularisation outside published ranges.** LR 1e-5 (published 1e-6 to 3e-6, including at 45M and 100M from scratch), KL 0.04 (published 0 to 1e-3), 256 samples per step. | A **val-only LR pilot** over {1e-6, 3e-6, 1e-5}; KL 1e-3; G=16 × 64 = 1,024 samples per update. | Zhang et al. 2512.07783 (100M); 2510.15990 (45M); DAPO; Skywork-OR1 |
| 4 | **Off-policy updates and biased normalisation.** 2 updates per rollout batch; group-std advantages, which overweight near-uniform groups (exactly this bimodal regime); a sequence-mean loss with a length bias. | **One update per batch**; **group mean / batch std** advantages; **constant-denominator token loss** (Dr. GRPO); clip-higher 0.28; truncated samples masked. | Skywork-OR1; Dr. GRPO; Tricks or Traps; ScaleRL |
| 5 | **A gate measuring the wrong thing.** The 0.20 mixed-group bar dates from v1, when every group was trained on. | Gate on **filter yield**, i.e. sampling cost per trained group, plus a **headroom gate** (pass@16 − pass@1) declared up front. | v3 Phase C lesson (post hoc, recorded before v4) |

**Bug check (done in Phase A).** `test_real_loop_raises_reward_on_a_learnable_rule` runs the
real loop end to end (sampling, scoring, advantages, update) with a stub reward the policy can
learn. Pass rate rises from about 0.5 to 1.0 within 40 steps, under both the v1 and the v4
objective. So the loop is not sign-broken, and rl-v2's flat reward came from the recipe or the
regime, not the code. (A side finding: with Adam and one update per batch, the global advantage
scale cancels. Only the *relative* weighting of groups differs between advantage modes.)

**Expected effect, stated before any run.** The closest analogues suggest a few points over the
parent on the deepest shapes the parent sometimes solves, and less over continued SFT. On a
converged parent, continued SFT should gain little, which is what makes the comparison
winnable. Parts of this design are post-hoc in origin: the filter-yield gate, the converged
parent and the LR range respond to v1–v3 results. But the data-v3 test suites have never been
used, so the confirmatory look stays clean.

## Questions and hypotheses

- **H4 (primary):** exact-outcome GRPO from the converged SFT parent raises held-out-shape
  correctness beyond (a) the parent and (b) time-matched continued SFT. The rule is in
  `reports/confirmatory_plan_v3.md`.
- **H1, H2:** unchanged from confirmatory plan v2, on `test_id` with the protocol v3 Phase B
  checkpoints.
- **Secondary (exploratory, one seed each, never part of the H4 decision):**
  - **v4-rft-s1:** on-policy rejection-sampling fine-tuning (expert iteration / RAFT). Identical
    to v4-grpo-s1 except advantage = reward with no baseline, and no KL. It isolates what GRPO's
    negative and baseline term adds. The literature places RFT within about 0–3 points of GRPO.
  - **v4-grpo-step-s1:** a bounded process reward. r = [exact answer] + 0.2 × (fraction of gold
    steps reproduced exactly as a prefix), and the filter admits 0–14 correct of 16 when rewards
    differ, so all-wrong groups with partial step credit can train.
    - **Why it can't be gamed:** gold traces are canonical, so the credit can't be inflated by
      padding, and every correct answer still outscores every wrong one.
    - **Not part of H4:** the primary metric stays exact answer.

## Phases

### Phase A (local, done)

**Code** (`src/microreason/rl/`):
- **Loss and advantages:**
  - advantage modes `group_std` (v1–v3), `group_mean` and `batch_std`, plus `objective: rft`;
  - `loss_aggregation: token_constant`;
  - `clip_eps_high`;
  - `mask_truncated`.
- **Rollouts:**
  - the pass-rate filter (`filter_min_correct` / `filter_max_correct`) under dynamic sampling;
  - the bounded `step_reward_weight` with `rollout.step_prefix_credit`;
  - micro-batched log-probs and gradient accumulation (`update_micro_batch`).
- **Gate and logging:**
  - `gate.min_filter_yield`, with the mixed-fraction rule now optional;
  - per-step `pass_rate`, `filter_yield`, `sampled_token_nll` (an entropy estimate) and
    `mean_step_credit`.
- **Overrides:** `--lr` for `rl grpo`; `--peak-lr` for `train`.

**Compatibility:** defaults reproduce the v1–v3 objective, and every earlier test still passes.

**Data:**
- **`RLPoolV3` options:** `family_keep` (declared keep probabilities), `id_prefix` and `purpose`.
- **Two new training-only pools:** each has its own root seed and id prefix, and excludes every
  frozen v3 evaluation record.

| Pool | Config | Records | Seed | Family mix | Used by |
| --- | --- | ---: | --- | --- | --- |
| SFT pool v4 | `configs/sft_pool_v4.yaml` | 5 M | 20261201 | natural | v2-sft-long: first 3 M of `sft_extra` |
| RL pool v4 | `configs/rl_pool_v4.yaml` | 4 M | 20261202 | 1op 3%, 2op 12%, 3op 25%, 4op 25%, 5op 20%, equations 15% | `gate`: gates; `grpo`: every RL arm's prompts; `sft_extra`: the control |

**Decision script:** `scripts/v4_decide.py` (subcommands `plateau`, `headroom`, `select`, `h4`).

**Session scripts:** `scripts/pod_v4_phase_b.sh` and `scripts/pod_v4_phase_c.sh`; setup
profiles `v4` and `v4c` in `pod_setup.sh`.

### Phase B (pod; held-out-VAL only; test suites sealed)

1. **Data.** Regenerate the v3 suites against their frozen hashes, then build the SFT pool v4.
   The RL pool v4 is built in the background, on CPU, during step 2.
2. **v2-sft-long** (`configs/sft_long_v4.yaml`). Continue p3-v2-sft for one epoch over 3 M fresh
   examples (46,875 steps), with the v3 SFT recipe and a cosine schedule from 5e-5 to 5e-6.
3. **Plateau curve.** Held-out-val greedy accuracy at 375 k, 750 k, 1.5 M and 3 M examples.
   - **Plateaued** if accuracy(3 M) − accuracy(1.5 M) < 1.0 point.
   - **Descriptive only:** the 3 M checkpoint is the parent either way.
   - **If it has not plateaued,** the report says continued SFT likely still has an advantage.
4. **Headroom gate.** v2-sft-long on `val_heldout_shape`, 16 samples at temperature 1.0.
   - **Rule:** pass@16 − sampled pass@1 ≥ 0.05, i.e. 5 points.
   - **If it fails,** RL is stopped under v4 and H4 stays undecided.
5. **KV-cache parity** (as in v3).
6. **Filter-yield gate.** 512 gate prompts, G=16.
   - **Rule:** at least 12.5% of groups pass the 2–14 filter, which caps sampling at about
     4 chunks per step; malformed ≤ 10%; pass@1 within [0.05, 0.95].
   - **If it fails,** stop.
7. **GRPO LR pilot.** 100 steps at each of {1e-6, 3e-6, 1e-5}, pilot seed 80. Select the highest
   held-out-val greedy accuracy; ties go to the lower LR.
8. **Control LR pilot.** Continued SFT at {5e-6, 2e-5}, each time-matched to the selected GRPO
   pilot run, selected by the same rule. **Both arms are tuned with the same val budget.**

**Output:** `reports/v4_phase_b/selection.json`, plus a report with the Phase C cost
re-estimated from the measured step time.

**Pilot results:** they are val-only and never touch a test suite. If no GRPO pilot beats
v2-sft-long on held-out-val, the report says so, and the owner decides whether Phase C is worth
running. That decision is recorded before any test suite is opened.

### Phase C (pod; confirmatory)

1. Rebuild the RL pool v4 (it must match Phase B's committed hash) and upload p4-v2-sft-long.
2. **For seeds 1–3:**
   - **v4-grpo-s{k}:** `configs/grpo_v4_seed{k}.yaml` at the pilot-selected LR, 300 steps.
   - **v2-extra-sft-s{k}:** time-matched to that GRPO run's measured training loop, at the
     pilot-selected LR.
3. **Secondary arms:** v4-rft-s1 and v4-grpo-step-s1, each at the GRPO LR.
4. **Val evaluations** of every checkpoint. **No checkpoint selection:** each run's final
   checkpoint is the one evaluated.
5. **Unseal once:** `test_heldout_shape` (greedy, plus 4 samples), `test_id` and
   `test_extrapolation` for every checkpoint; then H1/H2 on `test_id` with p3 checkpoints.
6. **Decision:** `scripts/v4_decide.py h4`.

## Fixed settings (primary arm)

| Setting | Value |
| --- | --- |
| Parent | p4-v2-sft-long (Phase B, 3 M-example continuation of p3-v2-sft) |
| Objective | on-policy GRPO, 1 update per rollout batch, clip [0.8, 1.28] |
| Advantage | (r − group mean) / std of all trained rewards in the step (`batch_std`) |
| Loss | Σ tokens / (completions × 256) (`token_constant`); truncated completions masked |
| KL | k3 to frozen parent, β = 1e-3 |
| Batch | 64 prompts × G=16 = 1,024 completions per update; chunks of 128 candidates, ≤ 8 chunks |
| Filter | keep a group iff 2 ≤ correct ≤ 14 of 16 |
| Sampling | temperature 1.0, ≤ 256 new tokens, KV-cache sampler, bf16 |
| LR | pilot-selected from {1e-6, 3e-6, 1e-5}; constant after 10 warmup steps |
| Steps | 300 (307 k trained completions per run) |
| Reward | binary exact-answer reward (the evaluation's scorer) |
| Seeds | 81, 82, 83 (each seed also fixes its prompt order); controls 91–93 |
| Evaluation | `eval-v2-cached`, greedy, 320 new tokens, show-work format |

**Compute matching.** The primary unit is training-loop wall time on the same pod: GPU-time is
the real cost, and it is the spec's "equal-cost". It is also the conservative choice, because
GRPO spends most of its time generating. The report gives FLOPs, updates, examples and tokens
for every arm. RFT, as a secondary arm, shares GRPO's rollout procedure exactly.

## Risks, declared now

- **Parent ceiling.** A converged parent leaves less headroom over itself (Yuan et al.: RFT gains
  shrink as SFT strengthens). The headroom gate tests this before any RL money is spent.
- **Depth limit.** If 5-operator failures come from the model's depth capacity, RL can't fix
  them. The filter then concentrates on 3–4 operators.
- **Pilot noise.** 100-step pilots can mis-rank LRs. Both arms get the same treatment.
- **Seeds.** Three seeds is the minimum the review recommends; the rule requires all of them to
  be positive.
- **What "held out" means.** As in v3, a held-out shape recombines parts seen in training, and
  Phase B v3 found no cost from shape novelty. `test_heldout_shape` mainly measures deep
  expressions with unseen structure.
- **Rollout records.** They are about 0.5–0.8 GB per run. Each run manifest keeps their sha256;
  the files are copied off the pod only on request (`PULL_ROLLOUTS=1`).

## Amendment 1 (2026-11-09 UTC, Phase B, before any pilot result)

On the Phase B pod a GRPO step took about 18 s: roughly 15 s to sample about 6,000 completions
(filter yield 0.21, so about 3 chunks per step) and about 2 s to update. The plan had assumed
8–10 s. The plan as written did not fit:
- **Phase B pilots** would have taken about 3.5 h, against a 4.5 h ceiling for the whole session.
- **Phase C** (8 runs × 300 steps) would have cost about $14, against its $9 cap.

The session was stopped 10 steps into the first pilot. That pilot was discarded unevaluated, and
no pilot or test result had been seen. Changes:

- **GRPO pilots:** 40 steps per LR, not 100. The same three LRs, seed and selection rule.
- **Control pilots:** still time-matched to the selected GRPO pilot run.
- **Phase C runs:** every RL arm runs **150 steps, not 300**: 153,600 trained completions per
  run, twice rl-v2's 76,800. The literature puts most RL gains in the first ~200 steps.
  Controls stay time-matched to their GRPO seed.
- **Phase C estimate:** about 7 h, about $7, within its $9 / 8 h cap.

**Already completed before the stop, and unchanged:**
- v2-sft-long: 80.7% on held-out-val, not plateaued, gaining 6.1 points over the last doubling.
- The headroom gate: passed, with pass@16 − pass@1 = 0.113.
- KV parity: passed, 23× faster.
- The filter-yield gate: passed at 0.213.

## Amendment 2 (2026-11-09 UTC, during Phase C seed 1, before any Phase C result)

GRPO steps on the Phase C pod take about 17–21 s. As written, the session could run to within
about 20 minutes of its 8 h ceiling. A budget stop inside a secondary arm would have ended the
session before the confirmatory test phase. The fix:
- **Secondary arms are non-blocking:** each one starts only if the time left covers its run plus
  a 50-minute test-phase allowance.
- **Incomplete arms are skipped:** an arm that does not complete is reported and skipped.
- **The test phase always runs** for every completed checkpoint.

**How it was applied:**
- **No edit to the running script:** bash reads scripts incrementally, so the running file was
  left untouched.
- **A copy and a watcher:** the amended script was copied to the pod as
  `scripts/pod_v4_phase_c_resume.sh`. A watcher on the pod stopped the original session when it
  printed its "secondary arms" marker, then started the copy with `FROM=secondary`. That
  happened after all six primary runs and their val evaluations.

**Unchanged:** the primary runs, the H4 rule, and the order in which test suites are opened
(after all training).

## Outcome (2026-11-09)

Phases B and C ran on one pod for about $10.47 (`reports/v4_phase_b.md`, `reports/v4_results.md`).

- **H4 not supported.** GRPO beats its parent on `test_heldout_shape` by +0.49 points
  [+0.09, +0.98], positive on all three seeds but far below +2.0. It loses to time-matched
  continued SFT by 1.06 points [−1.76, −0.53].
- **H1 and H2 supported** on `test_id`.
- **Exploratory:** GRPO beats continued SFT on `test_extrapolation` by +2.18 [+0.56, +3.88],
  because continued SFT erodes extrapolation (−2.4) while GRPO preserves it.
- **The data-v3 test suites are now used.** Any later evaluation on them is post hoc.
