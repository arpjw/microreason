# Outcome-reward RL vs SFT and compute-matched SFT-style baselines: evidence and design implications

Scope note: evidence is labelled **[LLM]** (large pretrained models, about 0.5B parameters and up, fine-tuned from public checkpoints) or **[SMALL/SCRATCH]** (small transformers trained from scratch on synthetic tasks). Most of the RL-vs-SFT literature is [LLM]. For many sources only the arXiv abstract was readable. Where a number comes from the full HTML text, the note says so.

## 1. Does RL improve greedy/pass@1 accuracy, or only sharpen the distribution? (Yue et al. 2025 and follow-ups)

### Takeaway
There is strong agreement that RLVR raises pass@1 a lot, mainly by moving probability onto solutions the model already finds at large k. Large-k coverage stays the same or shrinks. Pass@1 gains are biggest when pass@1 sits far below pass@k, which means headroom inside the support. Genuine expansion of the support has been reported only with very long RL (ProRL), with explicit exploration bonuses, or in controlled settings where RL data targets the "edge of competence". Even those cases require that pretraining or SFT already put some non-zero probability on the target structure.

### Cited Findings
- **[LLM] Yue et al. 2025 (NeurIPS 2025 Oral).** RLVR models "outperform their base models at small k (e.g., k = 1)", but base models reach higher pass@k at large k. Coverage and perplexity analyses indicate reasoning is "bounded by the base model". Six RLVR algorithms "perform similarly". Distillation, unlike RL, can add new reasoning patterns — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- **[LLM] Yue et al., full-text numbers (Qwen2.5 family, 7B to 32B).**
  - The base model overtakes RL "as k increases to the tens or hundreds". On Minerva at 32B, the base beats RL by about 9 points at k=128.
  - AIME24 (SimpleRLZoo, k=1024): 63.3% of problems solved by both, 13.3% base-only, **0.0% RL-only**.
  - MATH500 (k=128): 92.4% both, 3.6% base-only, 1.0% RL-only.

  Source: [arXiv HTML 2504.13837v5](https://arxiv.org/html/2504.13837v5)
- **[LLM] Yue et al., Table 4 (GRPO training dynamics).**
  - MATH500 pass@1/pass@256: base 34.5/96.2, then 74.4/97.2 at step 150, 75.4/96.0 at step 300, and 76.3/95.4 at step 450.
  - Omni-MATH-Train: base 9.9/67.2, rising to 42.5/64.3 at step 450.
  - So pass@1 improves greatly while pass@256 falls slowly. Most of the MATH500 pass@1 gain (+40 points) comes in the first 150 steps.

  Source: [arXiv HTML 2504.13837v5](https://arxiv.org/html/2504.13837v5)
- **[LLM] Yue et al., Table 3 (Omni-MATH-Test, pass@1/pass@256; base 10.2/69.1).** GRPO 25.1/68.3, PPO 26.8/69.2, ReMax 23.8/67.5, RLOO 28.1/69.2, Reinforce++ 28.0/69.7, DAPO 26.5/67.0. The algorithms land within about 4 points of each other on pass@1. A KL penalty of 0.001 gives similar pass@1 but much lower pass@128. RL raises the number of problems near accuracy 1.0 and also the number of problems at accuracy 0 — [arXiv HTML 2504.13837v5](https://arxiv.org/html/2504.13837v5)
- **[LLM] ProRL (Nemotron-Research-Reasoning-Qwen-1.5B).** Prolonged RL with KL control, reference-policy resets and a diverse task suite "can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling". Gains "correlate strongly with task competence of base model and training duration" — [arXiv 2505.24864](https://arxiv.org/abs/2505.24864)
- **[LLM] The Invisible Leash.** Across RLVR models, "the shrinkage of observed support outweighs its expansion": per question, more problems lose a verified success than gain one, even as pass@1 rises. Some RLVR models raise token-level entropy while lowering final-answer entropy — [arXiv 2507.14843](https://arxiv.org/abs/2507.14843)
- **[LLM] RL Squeezes, SFT Expands (ICLR 2026).** RL reduces the number of unique *incorrect* reasoning paths, while SFT increases the number of unique *correct* ones. RL steepens the decay of the step-level reasoning-graph distribution by about 2.5x, and SFT flattens it to about one third. This is offered as the explanation for the SFT-then-RL recipe — [arXiv 2509.21128](https://arxiv.org/abs/2509.21128)
- **[LLM] Reasoning with Exploration: An Entropy Perspective.** Adding an entropy-based term to the advantage, a one-line change, claims "significant gains on the Pass@K metric" even at very large K. No numbers were extracted — [arXiv 2506.14758](https://arxiv.org/abs/2506.14758)
- **[SMALL/SCRATCH] Zhang, Neubig, Yue 2025.**
  - Setup: 100M-parameter Qwen2.5-style model trained from scratch on 10B tokens of GSM-Infinite-style dependency-graph arithmetic. In-distribution is op=2–10, OOD-edge is op=11–14, OOD-hard is op=15–20.
  - **RL on in-distribution tasks improves pass@1 but not pass@128.** RL on edge-of-competence data (op=11–14) improves pass@128 on both OOD-edge and OOD-hard, by up to +42% pass@128.
  - Contextual transfer needs at least about 1% pretraining exposure to the new context, and then gives up to +60% pass@128.

  Source: [arXiv 2512.07783](https://arxiv.org/abs/2512.07783); [HTML](https://arxiv.org/html/2512.07783)
- **[SMALL/SCRATCH] "Can GRPO Help LLMs Transcend Their Pretraining Origin?"**
  - Setup: about 45M-parameter Llama-style model (4 layers, d=512) trained from scratch on synthetic string-operator tasks, then SFT on 2k examples and GRPO for 60 steps (1k prompts, 8 samples per prompt).
  - Theory: under KL-regularized RL, post-training success is f(Q)=aQ/(1−Q+aQ) with a=e^{1/β}, so **if Q=0 the gain is 0, and for small Q the gain is at most (a−1)Q**.
  - Empirically, when pretraining had 0% exposure to an OOD structure, GRPO gave 0 gain. With exposure, 3-step EM rose from 8% to 39% as the OOD pretraining ratio went from 2.5% to 33.3%. Compositional gains were up to +6 and +10 points. GRPO on an already-saturated ID task (about 98%) did not help and slightly hurt OOD.

  Source: [arXiv HTML 2510.15990v1](https://arxiv.org/html/2510.15990v1)
- **[SMALL/SCRATCH] Echo Chamber.** Models trained from scratch on open data mixtures, then fine-tuned with PPO, GRPO or Expert Iteration, "consistently converge toward a dominant output distribution" that amplifies a pretraining format. RL on simpler questions can still improve harder ones — [arXiv 2504.07912](https://arxiv.org/abs/2504.07912)

### Inferences
- In this literature, greedy gains appear where per-problem success sits at intermediate probabilities. RL pushes those problems toward 1.0. The relevant headroom is roughly (pass@k − pass@1) on the target distribution. In your gate pool that is 0.76 − 0.66 = 10 points at k=8. No source gives a formula for what fraction of that gap RL recovers.
- Expanding support (solving shapes with Q≈0) is unlikely in 60 to a few hundred GRPO steps on a tiny model. The f(Q) result in 2510.15990 is the most directly relevant theory for a ~43M from-scratch model.

### Gaps
- I did not extract the exact ProRL pass@k numbers or the exact entropy-advantage numbers (full text not read).
- I did not read the OpenReview discussions for Yue et al.

## 2. RL vs SFT on generalization (Chu et al. and rebuttals)

### Takeaway
Chu et al. report large OOD gains from RL and large *losses* from SFT on rule and visual variants. Their SFT was trained on fixed templates and ran into overfitting. Later work shows well-curated SFT (prompt diversity plus CoT) can match or beat RL. In both directions, the reported effect sizes depend heavily on how strong the SFT baseline is.

### Cited Findings
- **[LLM] Chu et al. 2025.** The model is Llama-3.2-Vision-11B. OOD rule-variant changes:

  | Task | RL | SFT |
  |---|---|---|
  | GeneralPoints-Language | **+3.5** (11.5→15.0) | −8.1 |
  | V-IRL-Language | **+11.0** (80.8→91.8) | −79.5 |
  | GeneralPoints-VL | +3.0 | −5.6 |
  | V-IRL-VL | +9.3 | −33.2 |

  On visual variants, RL gains +17.6 on GP-VL and +61.1 on V-IRL-VL.
  - The x-axis is training GFLOPs. RL from the raw base model failed. The initial SFT "stabilizes the model's output format".
  - RL could not recover OOD performance from an overfitted SFT checkpoint.

  Source: [arXiv HTML 2501.17161v2](https://arxiv.org/html/2501.17161v2)
- **[LLM] Debunk the Myth of SFT Generalization.** On Sokoban and General Points, SFT's failure comes largely from fixed instruction templates. SFT with prompt diversity plus CoT generalizes across instruction and difficulty shifts, including arithmetic with OOD values, "matching or surpassing RL baselines" — [arXiv 2510.00237](https://arxiv.org/abs/2510.00237)
- **[LLM] Retaining by Doing.** On Llama and Qwen across instruction following, knowledge and arithmetic, RL forgets less than SFT while getting "comparable or higher target task performance". The cause is attributed to on-policy data, not to the KL term or the advantage estimator — [arXiv 2510.18874](https://arxiv.org/abs/2510.18874)
- **[LLM, synthetic arithmetic] Zipping the Thought.**
  - Setup: Qwen2.5 0.5B–14B and Llama 1B–8B on mod-23 arithmetic chains.
  - Finding: "SFT cannot decompose reasoning steps below the compression granularity of the training data". After SFT, odd-op OOD tasks were near chance (1/23). After GRPO on odd-op tasks (op=9–15), models acquired decomposition.
  - No compute-matched SFT control was run for the RLVR experiments.

  Source: [arXiv HTML 2605.28008](https://arxiv.org/html/2605.28008)
- **[LLM] RL for compositional generalization (2026 preprint).** On SCAN, COGS, GeoQuery and CFQ, pass@1 was SFT 77.90 vs GRPO-C 83.42 — [arXiv 2605.04920](https://arxiv.org/pdf/2605.04920). This is from search-result summaries only. Not verified.
- **[SMALL/SCRATCH] Course project.** On n-digit addition with 125M–360M models, "SFT on problem answers" gave the best generalization. A possible GRPO advantage appeared only for the larger model with cold-start CoT — [CS224R report](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report-2.pdf). This is unreviewed and was seen only in a search summary.

### Inferences
- The famous RL-over-SFT gaps come from SFT runs that degrade OOD, for example by overfitting to templates. Your continued-SFT control trains on *fresh, unlimited, correct* traces, so it does not have that failure mode. A Chu-style gap should not be expected.
- RL's plausible structural advantage on held-out shapes appears where the SFT traces cannot express the needed procedure, as with compressed steps in 2605.28008. It does not appear where the oracle trace generator already covers the procedure.

### Gaps
- I found no study on a from-scratch model under 100M parameters that directly compares GRPO with a *time-matched, fresh-data* continued-SFT control on held-out arithmetic shapes.

## 3. Expert iteration / RFT / STaR / RAFT vs PPO/GRPO

### Takeaway
On math, expert iteration and rejection-sampling fine-tuning roughly match PPO-class RL. GRPO pulls ahead only modestly and late, by a few points. Its edge comes mostly from filtering out prompts with no learning signal and from keeping entropy higher. RFT gains shrink as the SFT model gets stronger and produces fewer distinct correct paths.

### Cited Findings
- **[LLM] Havrilla et al. 2024 (Llama-2 7B/13B, GSM8K, SFT init).** Values are maj@1 and pass@96, each as 7B/13B:

  | Method | maj@1 | pass@96 |
  |---|---|---|
  | SFT | 0.41/0.48 | 0.72/0.84 |
  | **Expert Iteration** | **0.48/0.53** | 0.80/0.88 |
  | Sparse PPO | 0.44/0.51 | – |
  | Dense PPO | 0.43/0.50 | – |
  | Sparse ORM PPO | 0.46/0.51 | – |

  - **EI beat or matched every PPO variant.**
  - PPO peaked after about 60k rollouts. EI used roughly 10x more samples, but both took about a day of wall time because PPO's memory limited throughput.
  - Without SFT, EI with K=4 samples per question matched PPO's sample complexity at an accuracy only a few points lower.
  - Pass@96 saturates early in RL, so RL mainly improves maj@1. The authors attribute the small gains to limited exploration and deterministic dynamics.
  - Compute was *not* matched. Comparisons are by rollouts and wall time.

  Source: [arXiv HTML 2403.04642](https://arxiv.org/html/2403.04642)
- **[LLM] Yuan et al. 2023 (RFT, GSM8K greedy, k=100 samples at T=0.7).** SFT→RFT:

  | Model | SFT | RFT | Gain |
  |---|---|---|---|
  | LLaMA-7B | 35.9 | 41.7 | +5.8 |
  | LLaMA-13B | 43.0 | 49.1 | +6.1 |
  | LLaMA-33B | 54.6 | 54.5 | −0.1 |
  | LLaMA2-7B | 41.6 | 47.5 | +5.9 |
  | LLaMA2-13B | 50.0 | 54.8 | +4.8 |

  - Pooling samples from multiple models (RFT-U13B) gives up to +13.4 for 7B, and −0.3 / −0.9 for 65B / 70B.
  - The key driver is the **number of distinct correct reasoning paths**: 1.17 per question at k=1 vs 5.25 at k=100 for 7B. The 33B model produced only 2.78, and RFT did not help it.
  - With k=3, RFT gives about +2 points.

  Source: [arXiv HTML 2308.01825](https://arxiv.org/html/2308.01825)
- **[LLM] Minimalist approach (Xiong et al. 2025).** Average over Math500, Minerva and Olympiad, avg@16 at T=1.0:

  | Method | Qwen2.5-Math-7B | LLaMA-3.2-3B-Instruct |
  |---|---|---|
  | Base | 23.6 | 13.1 |
  | RAFT | 52.3 | 25.9 |
  | RAFT++ | 56.1 | 27.6 |
  | GRPO | 56.3 | 28.4 |
  | PPO | 52.5 | 26.9 |
  | Reinforce-Rej | 56.4 | 28.5 |

  - **RAFT++ (rejection sampling with importance weighting and clipping) is within 0.2–0.8 points of GRPO.** It converges faster early, then plateaus around iteration 100 as entropy collapses, and GRPO overtakes it.
  - GRPO's gain comes mainly from dropping prompts whose samples are all wrong. Reinforce-Rej, which drops all-wrong and all-correct groups, matches GRPO.
  - Compute was not explicitly matched. Shared settings: 1024 prompts per iteration, 4 samples per prompt.

  Source: [arXiv HTML 2504.11343](https://arxiv.org/html/2504.11343)
- **[LLM] Setlur et al. 2024.** Fine-tuning on correct solutions from stronger models gives "modest" gains. Fine-tuning on the learner's *own* correct solutions "doubles the efficiency" of the same problems. Positives-only self-training can amplify spurious correlations, with flat or inverse scaling. Adding negatives with per-step credit, which is equivalent to advantage-weighted RL, is worth about 8x more synthetic data — [arXiv 2406.14532](https://arxiv.org/abs/2406.14532)

### Inferences
- In this literature, the natural "fair" self-improvement baseline for GRPO is iterated RFT/EI on the model's own correct samples, using the same prompts and the same number of rollouts. Expected separation from GRPO is 0–3 points in [LLM] results.
- Your continued-SFT control is a different kind of baseline. Fresh correct oracle traces work like unlimited distillation from a perfect teacher, which is the regime where Yue et al. say distillation *can* add capability. Losing to it by 6.3 points is consistent with this literature, not an anomaly.

### Gaps
- I did not fetch exact numbers for ReST-EM (Singh et al. 2023, arXiv 2312.06585), the original RAFT (2304.06767), or STaR. From memory only, not verified this session: ReST-EM on PaLM-2 beat fine-tuning on human data for MATH and APPS, with gains growing with model size.

## 4. Compute matching, on-policy vs ground-truth data, and expected effect sizes; implications for design

### Takeaway
No surveyed paper does strict FLOP-matching between RL and an SFT control. Most match rollouts, steps or wall time, or report compute on the x-axis (Chu: GFLOPs). Matching on wall time strongly favours SFT, because RL spends most of its time generating. Reported greedy gains of RL over a strong SFT/RFT model on math are usually **+2 to +7 points** [LLM], and they shrink as the SFT model gets stronger. A ≥2-point win over a fresh-data continued-SFT control is at the edge of what the literature supports, unless the evaluation targets something oracle SFT traces do not teach.

### Cited Findings
- Compute accounting in practice:
  - Havrilla compares by rollouts and wall time, noting that PPO and EI both took about a day — [2403.04642](https://arxiv.org/html/2403.04642)
  - Chu plots against training GFLOPs — [2501.17161](https://arxiv.org/html/2501.17161v2)
  - Zhang et al. split a fixed budget between mid-training and RL (β = 0.2, 0.5, 0.8; full mid-training = 1B supervised tokens vs full RL = 100 steps × 1024 batch). Mid-training plus RL beat RL alone by +10.8% on OOD-hard. Light-RL (β=0.2) gave the best pass@1 on OOD-edge, and Heavy-RL helped OOD-hard — [2512.07783](https://arxiv.org/html/2512.07783)
  - Zipping the Thought matches compute only for SFT data repetition (384k×1 epoch vs 6k×64 epochs) — [2605.28008](https://arxiv.org/html/2605.28008)
- On-policy data:
  - Own correct samples are about 2x as sample-efficient as stronger-model samples — [2406.14532](https://arxiv.org/abs/2406.14532)
  - On-policy data is the cause of RL's lower forgetting — [2510.18874](https://arxiv.org/abs/2510.18874)
  - Distinct self-generated paths drive RFT gains — [2308.01825](https://arxiv.org/html/2308.01825)
- Effect sizes over strong SFT [LLM]:
  - EI over SFT: +5 to +7 maj@1; PPO over SFT: +2 to +5 — [Havrilla](https://arxiv.org/html/2403.04642)
  - RFT over SFT: about +5–6 for 7B–13B, and about 0 when SFT is already stronger (33B) — [Yuan](https://arxiv.org/html/2308.01825)
  - GRPO over RAFT++: +0.2 to +0.8 — [Xiong](https://arxiv.org/html/2504.11343)
  - Very large gains over *base* models (MATH500 34.5→76.3) come from base models with large pass@1/pass@k gaps (34.5 vs 96.2) — [Yue](https://arxiv.org/html/2504.13837v5)
- [SMALL/SCRATCH] RL on in-distribution difficulty improves pass@1 only. To get gains beyond that, the RL data must sit at the edge of competence — [2512.07783](https://arxiv.org/html/2512.07783). GRPO helps only when Q>0 on the target structure, and gains scale with Q — [2510.15990](https://arxiv.org/html/2510.15990v1)

### Inferences (design implications for the pre-registration)
1. **Headroom.** pass@1 is 0.66 and pass@8 is 0.76. In the literature, RL mostly converts within-support probability mass into greedy correctness, so 10 points is a loose ceiling at k=8, and larger k gives a somewhat larger ceiling. The [LLM] results suggest recovering roughly a third to a half of such a gap. That implies an expected +3 to +5 over the parent and **less over the continued-SFT control**, because continued SFT also sharpens. This is my estimate, not a sourced number. Gate on pass@k−pass@1 measured on the *held-out shapes*, not only on the gate pool.
2. **Control fairness.**
   - Wall-time matching hands SFT many more gradient tokens, because GRPO time goes to generation.
   - A defensible "equal-cost" scheme pre-registers one primary unit and reports the others. Options are total FLOPs (generation plus backward), optimizer steps × tokens, or problems seen.
   - Report the continued-SFT control as an **upper-bound oracle-distillation control**, not as the matched control.
   - Make the primary matched control **RFT/EI**: the same prompts, the same number of samples, fine-tuning on the model's own correct samples. This follows Havrilla, Yuan and Xiong. It isolates "outcome-RL vs on-policy SFT", which is what the literature actually contests.
3. **Where RL can plausibly beat oracle SFT.** RL has room only where the oracle trace generator does not teach the needed behaviour:
   - held-out shapes that need a different decomposition, as with compressed steps in 2605.28008;
   - edge-of-competence shapes with Q>0 but low, as in 2512.07783 and 2510.15990;
   - inference-time robustness to the model's own errors, i.e. exposure bias.

   Training GRPO on edge-of-competence shapes, and dropping all-correct and all-wrong groups (Reinforce-Rej / DAPO-style), follows the conditions under which gains have been reported.
4. **Hybrid arm.** Consider an arm of continued SFT followed by GRPO, versus continued SFT alone at matched total budget. This mirrors the "SFT expands, RL squeezes" recipe [2509.21128](https://arxiv.org/abs/2509.21128) and the mid-training-plus-RL allocation result [2512.07783](https://arxiv.org/html/2512.07783). It is more winnable than RL replacing SFT.
5. **Statistics.** With 2 seeds and expected effects of 2–5 points, pre-register per-problem paired tests across seeds. Also track per-question support shrinkage, following the Invisible Leash method [2507.14843](https://arxiv.org/abs/2507.14843).

### Gaps
- I found no paper that directly tests exposure bias, i.e. teacher-forced ground-truth traces vs on-policy samples, at the scale of a from-scratch model under 100M parameters.
- I found no source that quantifies RL greedy gain as a function of the (pass@k − pass@1) gap for a strong SFT model. Point 1 above is an extrapolation.
- Exact per-band pass@1 numbers in 2512.07783 are only in its figures and were not extracted.
