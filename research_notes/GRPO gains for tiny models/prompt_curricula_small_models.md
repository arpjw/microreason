# Prompt selection, difficulty curricula and prolonged training for outcome-reward RL (small models, synthetic tasks)

Scope note: almost all quantitative evidence on prompt filtering, group size and prolonged RL comes from pretrained LLMs of 1.5B to 32B parameters (Qwen2.5, Qwen3, DeepSeek-R1-Distill). Those results are flagged **[LLM-only]**. Evidence at or near the ~43M scale comes from three sources: a 45M from-scratch GRPO study (arXiv 2510.15990), a single-layer-transformer theory paper (arXiv 2601.15158) and a 19M community Countdown model.

## Offline pass-rate pre-filtering (thresholds and how much it helps)

### Takeaway
Every major recipe drops prompts the current policy always solves, and most also drop prompts it never solves. Pass rates are typically estimated from 8 to 16 rollouts. Band filters near 0.5 (for example 0.2–0.8 or 0.3–0.7) beat no filtering, but the measured gain from offline filtering alone is small (about +2 points). Online or re-estimated filtering helps more, and filtering out everything except the hardest prompts hurt POLARIS.

### Cited Findings
- **POLARIS [LLM-only, Qwen3-4B, R1-Distill-Qwen-7B].** Difficulty is the pass rate over 8 rollouts from the model being trained, at temperature 0.6. Only 8/8-solved prompts are removed: DeepScaleR's 40K prompts became 26K, and adding AReaL gave a 53K pool, which a further pass cut to 30K for Qwen3-4B. A more aggressive filter that removed every prompt with a pass rate above 4/8 (leaving 19K) hurt training. The authors' takeaway is that a balanced, slightly hard-skewed difficulty distribution works best. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris)
- POLARIS reported that the DeepScaleR recipe, used without recalibrating difficulty for Qwen3, gave only marginal gains or even declines. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris)
- **Results of POLARIS (avg@32).** Polaris-4B scored 81.2 vs 73.8 on AIME24 and 79.4 vs 65.6 on AIME25, against base Qwen3-4B. Polaris-7B scored 72.6 vs 55.0 and 52.6 vs 39.7, against base R1-Distill-7B. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris)
- **Skywork-OR1 [LLM-only, 7B and 32B].** Before training, it removes prompts on which the base model's correctness rate is exactly 0 or 1. At the start of each stage, it also drops prompts the actor fully solved in the previous stage, and its rejection sampling keeps only groups with non-zero advantage in each batch. Overall gains were +15.0 points (57.8 to 72.8) for 32B and +13.9 points (43.6 to 57.5) for 7B, averaged over AIME24, AIME25 and LiveCodeBench. The paper does not isolate how much of that gain comes from filtering. — [Skywork-OR1 HTML](https://arxiv.org/html/2505.22312v2); [abstract](https://arxiv.org/abs/2505.22312v1)
- **Offline vs online filtering, controlled comparison [LLM-only, Qwen2.5-3B SFT].** Offline curation and offline scheduling with a (0.2, 0.8) band gave at most +2.1 points over plain GRPO. Balanced online filtering beat both offline methods consistently. Using an external model to estimate difficulty beat using the initial SFT model on average, though results varied by benchmark. — [Online Difficulty Filtering, arXiv 2504.03380](https://arxiv.org/pdf/2504.03380v2)
- **AdaRFT [LLM-only, Qwen2.5].** Difficulty comes from dataset annotations or from the model's success rate over n=128 attempts per problem. A moving target difficulty is updated as T' = clip(T + η·tanh(α·(R_avg − β)), d_min, d_max), and each batch draws the problems closest to T. The method reduced RFT training time by up to 2×, with the largest gains on imbalanced data. Reported hyperparameters (from a secondary summary) are target reward β=0.5, sensitivity α=2, step size η=50 and batch size 1024. — [AdaRFT arXiv 2504.05520](https://arxiv.org/html/2504.05520v3); [Moonlight review](https://www.themoonlight.io/en/review/efficient-reinforcement-finetuning-via-adaptive-curriculum-learning)
- **Learnability-based selection (SFL; Foster and Foerster, arXiv 2502.12272) [LLM-only].** Questions are scored by p(1−p), the Bernoulli variance of success, and the top-K are kept in a buffer. This reportedly gave a 3.3× speedup in training steps to reach baseline accuracy with VinePPO on GSM8K. That figure comes from a secondary summary and was not verified in the paper. — [Moonlight review](https://themoonlight.io/review/learning-to-reason-at-the-frontier-of-learnability)
- **MoPPS (arXiv 2507.04632)** predicts each prompt's success rate online with a Beta posterior and Thompson sampling, without extra LLM rollouts, and selects prompts near a target rate of about 0.5. — [MoPPS](https://arxiv.org/pdf/2507.04632)
- **Kimi k1.5** does not filter by a hard pass-rate band. Its prioritized sampling tracks each problem's latest pass rate and gives lower-pass-rate problems a higher sampling probability. This description comes from a secondary citation. — [arXiv 2505.17652](https://arxiv.org/pdf/2505.17652v2)
- **Light-R1:** no source with its filtering thresholds was found (see Gaps).

### Inferences
- At the current SFT distribution (56% all-correct, 24% all-wrong, 19.5% mixed at G=8), drawing prompts uniformly wastes about 80% of rollouts. Since prompts are unlimited, offline filtering is cheap: score a large candidate pool with 8–16 samples and keep 0 < p < 1, or a 0.2–0.8 band. This should lift the mixed fraction in each batch from about 20% to close to 100% at the start of training.
- Depth is a strong free proxy for difficulty (95% accuracy at 1 operator, about 37% at 5). Up-weighting depths 3–6 and the hard tail of depth 2 is an AdaRFT-style annotation-based curriculum that needs no scoring rollouts. Per-prompt scoring still matters, though, because the all-or-nothing pattern means pass rates vary a lot within a depth.
- The POLARIS finding that removing every prompt above a 4/8 pass rate hurt suggests that keeping some moderately easy prompts (p around 0.5–0.9) is safer than keeping only the hard ones.

### Gaps
- Light-R1's thresholds were not retrieved. The original DAPO, Skywork-OR1 and SFL papers were not checked for the exact number of scoring rollouts.
- No source measured offline pass-rate filtering on a model under 100M parameters.

## Online difficulty curricula, re-filtering, and zero-variance prompts

### Takeaway
Online filtering, which re-samples until a batch is full of prompts with 0 < p < 1 or inside a band, is the best-supported technique. With a Qwen2.5-3B SFT model it gave +4 points on average and up to +10 points on AIME over plain GRPO, and the narrower 0.3–0.7 band worked best. Filtering only easy prompts or only hard prompts did worse than no filtering. All-correct and all-wrong groups give zero gradient under standard GRPO. RL-ZVP recovers signal from them through entropy-guided advantage shaping and reported +4 points.

### Cited Findings
- **Online Difficulty Filtering (arXiv 2504.03380) [LLM-only, Qwen2.5-3B, SFT cold start on 1.1K R1-distilled solutions].**
  - Setup: GRPO with 16 rollouts for each of 16 prompts per step, on NuminaMath.
  - Bands tested: (T_low, T_high) ∈ {(0,1), (0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6)}.
  - Averaged over five benchmarks, plain GRPO reached 26.3%. The (0,1) filter, which removes only all-correct and all-wrong groups, reached 27.3%, and (0.3,0.7) exceeded 30%. Gains on AIME reached 10 points.
  - Skewed filters (removing only easy prompts or only hard ones) reached 24.9–25.9%, worse than plain GRPO.
  - With Qwen2.5-7B, (0.3,0.7) was again best, with +3 points on AIME and +5 points on AMC.
  - Balanced filtering beat plain GRPO's best reward in less than 50% of the steps and also won in wall-clock time. The (0.4,0.6) band needed many more rollouts to fill each batch, which made overall training slower.
  - Harder benchmarks and harder MATH500 levels (3–5) gained the most.
  - Theory: a lower bound on learnability is proportional to the Bernoulli variance p(1−p), which motivates a band around 0.5.
  - Source: [arXiv 2504.03380](https://arxiv.org/pdf/2504.03380v2)
- **DAPO dynamic sampling [LLM-only].** It over-samples and drops prompts whose group accuracy is exactly 0 or 1, continuing until the batch is full of non-degenerate groups. As the model improves, more groups are filtered, which raises rollout cost. — [DAPO arXiv 2503.14476](https://arxiv.org/pdf/2503.14476v2); [reading note](https://jianyuh.github.io/dapo/2025/05/01/DAPO.html); [Towards AI summary](https://towardsai.net/?p=43227)
- **POLARIS online updating.** Each sample's accuracy is updated from rollout rewards during training, and at the end of each training phase samples with accuracy above 0.9 are removed. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris); [Seed blog](https://seed.bytedance.com/blog/seed-s-open-source-rl-recipe-makes-4b-model-rival-235b-on-math-reasoning)
- **Skywork-OR1** removes prompts solved with correctness 1 at each stage boundary and keeps only groups with non-zero advantage in each batch. — [Skywork-OR1](https://arxiv.org/html/2505.22312v2)
- **RL-ZVP, "No Prompt Left Behind" (arXiv 2509.21880, ICLR 2026) [LLM-only, Qwen3-1.7B-Base and 8B-Base].**
  - Zero-variance prompts made up 30% to 99% of each batch during training.
  - RL-ZVP applies entropy-guided, token-level advantage shaping to those prompts: it rewards correct all-correct groups and penalizes all-wrong ones. Prompts with non-zero variance use standard GRPO.
  - Across six math benchmarks it improved Acc@8 by 4.00 points and Pass@8 by 4.28 points on average over GRPO, with peaks of 8.61 points (v1) or 8.66 points (v2, AIME25). The two versions report slightly different peaks.
  - It beat filtering baselines even when those were given 3×–5× more rollouts.
  - Source: [arXiv 2509.21880](https://arxiv.org/abs/2509.21880v2); [ICLR poster](https://iclr.cc/virtual/2026/poster/10007755)
- **CDAS (arXiv 2505.17652)** criticizes DAPO's rollout cost and selects prompts using accumulated pass-rate history rather than a single step's pass rate. — [arXiv 2505.17652](https://arxiv.org/pdf/2505.17652v2)
- **Theory on single-layer transformers (arXiv 2601.15158, Tel Aviv University).** Outcome-reward policy gradient provably learns an iterative, step-by-step graph-traversal algorithm that extrapolates to longer chains. This holds only when the training distribution puts enough mass on simple examples with fewer steps. When that mass vanishes, policy-gradient learning becomes infeasible. The paper corroborates this on synthetic data and on real LMs. — [arXiv 2601.15158](https://arxiv.org/pdf/2601.15158v2)

### Inferences
- For a 43M model, the theory and the 45M study below both argue against a curriculum of hard prompts only. Keep easier, shallower examples in the mix so the step-by-step algorithm is reinforced, while weighting the batch toward the mixed band.
- Because prompts are unlimited, DAPO's rollout-cost objection mostly disappears: generating and scoring extra candidates is cheap at 43M parameters. A practical design is a streaming filter that scores fresh prompts at G=8 and keeps those with 1/8 ≤ correct ≤ 7/8 (or 2/8–6/8), refreshed continuously. Prompts never need to be repeated, so there is no stale pass-rate problem.
- RL-ZVP is worth considering if filtering cannot keep the mixed fraction high. However, the reward from all-correct groups may simply reinforce easy depth-1 behavior. The +4 points were measured on Qwen3 models of 1.7B and 8B parameters, not on tiny models.

### Gaps
- No curriculum or filtering ablation on a from-scratch model under 100M parameters was found.
- How often to re-filter (in steps) was not quantified beyond per-stage (Skywork, POLARIS) and per-batch (DAPO, online filtering).

## Group size, larger groups for rare successes, and pass@k objectives

### Takeaway
Typical recipes use G=8 (POLARIS, the 45M study) or G=16 (Skywork-OR1, online filtering, the BroRL baseline). Skywork found no entropy difference across G=4/8/16, while BroRL showed that raising G from 16 to 512 restarted progress on a 1.5B model that had plateaued. Pass@k-style objectives exist (PKPO; Chen et al. 2025), but a November 2025 critique argues pass@k is a diagnostic rather than a good objective. No controlled 8/16/32/64 sweep was retrieved.

### Cited Findings
- **Skywork-OR1.** G = 4, 8 and 16 produced no significant differences in entropy dynamics. Rollout batch sizes of 16, 32 and 64 also gave similar entropy dynamics, and Section 5.2 reports that larger batch sizes improve test performance. The baseline uses G=16. — [Skywork-OR1](https://arxiv.org/html/2505.22312v2)
- **BroRL (NVIDIA, arXiv 2510.01180) [LLM-only, 1.5B].**
  - Starting point: a ProRLv2 checkpoint that had plateaued after 3,000 RL steps.
  - Raising rollouts per prompt from 16 to 512 moved the math score from a stagnating 62.08 (continued ProRL peak) to 62.85 after 134 more steps.
  - Within 98.1 hours, BroRL had beaten ProRL's final scores on all metrics, about 35 hours sooner.
  - Generation throughput rose from 36.5 to 72.4 samples/s, because the workload became compute-bound.
  - Source: [BroRL](https://arxiv.org/html/2510.01180v1); [NVIDIA blog (promotional)](https://developer.nvidia.com/blog/breaking-through-rl-training-limits-with-scaling-rollouts-in-brorl)
- **Online filtering with G=16** worked with a 0.3–0.7 band. The narrower 0.4–0.6 band is effectively a resolution limit at small G and needed many more rollouts per filled batch. — [arXiv 2504.03380](https://arxiv.org/pdf/2504.03380v2)
- **RL-ZVP** beat filtering baselines that were given 3–5× more rollouts, which suggests that extracting signal from zero-variance groups can substitute for larger groups. — [arXiv 2509.21880](https://arxiv.org/abs/2509.21880v2)
- **PKPO (Walder and Karkhanis, May 2025)** derives unbiased pass@k gradient estimators that treat the k samples jointly. **Chen et al. (August 2025)** derive an analytical pass@k advantage function. A common practice is to anneal k from high toward 1. — [Yu, arXiv 2511.16231](https://www.alphaxiv.org/abs/2511.16231); [Emergent Mind topic (secondary)](https://www.emergentmind.com/topics/pass-k-policy-gradients)
- **Critique (arXiv 2511.16231, November 2025).** Pass@k is a useful diagnostic of exploration but an unsuitable direct objective. As the policy concentrates probability mass, the gap between pass@k and pass@1 shrinks ("exploration collapse"). — [arXiv 2511.16231](https://arxiv.org/html/2511.16231v1)

### Inferences
- The probability that a group is mixed is 1 − p^G − (1−p)^G. For a prompt with p=0.05, that is about 34% at G=8, 56% at G=16, 81% at G=32 and 96% at G=64. Larger groups mainly help on the hard tail, such as depth 5–6 prompts with low pass rates. At 43M parameters, rollouts are cheap, so G=16–32 restricted to hard prompts is affordable.
- With all-or-nothing outcomes, many all-wrong groups probably have p near 0 rather than slightly positive, and larger G cannot rescue those. Per the 45M study below, GRPO cannot create a skill the model has no prior exposure to.

### Gaps
- No controlled 8/16/32/64 comparison with held-out accuracy was retrieved. Exact gains for PKPO and Chen et al. were not verified from primary sources.

## Prolonged RL (ProRL, BroRL): how long until gains appear

### Takeaway
ProRL trained a 1.5B distilled model for thousands of steps with KL control and reference-policy resets, and BroRL found a plateau around 3,000 steps that was broken by scaling rollouts. These are LLM results. At the other extreme, the 45M from-scratch study ran GRPO for only 60 steps on 1,000 samples and still measured gains where prior exposure existed. For tiny models, step count seems less limiting than whether the model already has some competence on the target.

### Cited Findings
- **ProRL (arXiv 2505.24864) [LLM-only].** Nemotron-Research-Reasoning-Qwen-1.5B was trained from DeepSeek-R1-Distill-Qwen-1.5B on 136K diverse problems, using KL divergence control, reference policy resetting and a diverse task suite. Reported average pass@1 gains were +14.7% on math, +13.9% on code, +54.8% on logic puzzles, +25.1% on STEM and +18.1% on instruction following. A secondary summary gives different figures (for example +15.7% on math). Improvements in the reasoning boundary correlate with the base model's task competence and with training duration. — [ProRL arXiv](https://arxiv.org/html/2505.24864v1); [HF papers](https://huggingface.co/papers/2505.24864)
- **BroRL.** A 3,000-step ProRLv2 checkpoint had plateaued, and continued step scaling peaked and then degraded. — [BroRL](https://arxiv.org/html/2510.01180v1)
- **Skywork-OR1-7B** ran 2,700 steps on 32 H800 GPUs. In the multi-stage setup, stage I (8K context) switched to stage II (16K) at step 540. — [Skywork-OR1](https://arxiv.org/html/2505.22312v2)
- **POLARIS** observed that the robust and exploration temperature zones shift after about 800 RL steps, which motivated raising the temperature at each stage. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris)
- **TinyZero-style 19M Countdown model (community).** GRPO ran for about 500 steps from a pretrained checkpoint and moved accuracy from 31.6% to 34.4% (+2.8pp) on 250 held-out problems. The author calls the effect modest. — [HF model card](https://huggingface.co/simak31/tinyzero-countdown-19m)

### Inferences
- Gains of several points on held-out data at small scale have appeared within hundreds of steps (60 steps in the 45M study, about 500 in the 19M model). Evidence for long-horizon gains (thousands of steps) exists only for models of 1.5B parameters and up.
- ProRL's finding that boundary gains correlate with base competence suggests prolonged RL will help most at depths 2–4, where the model has partial competence.

### Gaps
- The exact ProRL step count and the number of samples needed before gains appear were not verified from the paper. No source gives a sample-count curve for models under 100M parameters.

## Results from TinyZero, Mini-R1 and RL on tiny or from-scratch models

### Takeaway
The closest analogue is "Can GRPO Help LLMs Transcend Their Pretraining Origin?" (arXiv 2510.15990). It used about 45M-parameter Llama-style models trained from scratch on synthetic string tasks, with GRPO at 8 samples, 60 steps and a learning rate of 3e-6. GRPO improved out-of-distribution depth, length, token or composition only when pretraining had some exposure to the target, and never extrapolated to unseen depths. TinyZero reports that Qwen2.5-0.5B fails at Countdown and that about 3B is needed.

### Cited Findings
- **arXiv 2510.15990 (Ni, Tan, Liu, Li, Chen; UNC and ASU).**
  - Model: hidden size 512, FFN 1376, 4 layers, 8 heads. The paper says "around 4545 million" parameters, which is likely a typo for about 45M.
  - Pipeline: pretraining on about 67M samples mixing in-distribution and out-of-distribution data, then SFT on 2,000 in-distribution examples, then GRPO on 1,000 samples at learning rate 3e-6, batch 64, 8 samples per prompt, KL 0.005 and 60 steps.
  - Depth: without 3-step data in pretraining, the model completely fails at 3-step tasks even after GRPO. Adding 3-step data raised 3-step exact match by 8% to 39% (as the out-of-distribution ratio in pretraining went from 2.5% to 33.3%). GRPO on 2-step data improved 3-step accuracy, and GRPO on 1-step data did not. The authors conclude that GRPO does not extrapolate to unseen depths.
  - Length: GRPO was ineffective with zero exposure to longer inputs, and peaked at +21% with 12.5% out-of-distribution pretraining data.
  - Composition: with only single operations in pretraining, composites scored 0% and GRPO did not help. With 2.5–5% composite data, composites reached about 75%, and GRPO added +6 to +10 points.
  - Token representation: GRPO gave up to +30%, peaking at 25% out-of-distribution pretraining data.
  - In-distribution GRPO on already-saturated tasks (98% accuracy) did nothing.
  - Source: [arXiv 2510.15990](https://arxiv.org/html/2510.15990v1)
- **TinyZero.** The README states that Qwen2.5-0.5B base fails to learn reasoning on Countdown. Reasoning behaviors such as search and self-verification appear at about 3B parameters. — [TinyZero mirror](https://github.com/vra/TinyZero); [starlog summary](https://starlog.pages.dev/articles/developer-tools/jiayi-pan-tinyzero)
- **19M Countdown model.** Pretraining used about 380M tokens (a ratio of about 20 tokens per parameter). SFT reached 0% generation accuracy because of template dilution, so GRPO was run directly on the pretrained checkpoint, giving +2.8pp. An earlier 116M-parameter run on only 150M tokens generalized poorly. — [HF model card](https://huggingface.co/simak31/tinyzero-countdown-19m)
- **Teaching Arithmetic to Small Transformers (Lee et al., ICLR 2024; supervised, not RL).** Small from-scratch models learn arithmetic, and chain-of-thought or reversed outputs improve sample efficiency. Length generalization stays poor even for one held-out digit length. — [arXiv 2307.03381](https://arxiv.org/pdf/2307.03381)
- **Outcome-based RL theory (arXiv 2601.15158).** On single-layer transformers, policy gradient learns a vertex-by-vertex algorithm that extrapolates to longer chains when the training data includes enough simple examples. — [arXiv 2601.15158](https://arxiv.org/pdf/2601.15158v2)

### Inferences
- The 45M study is the most transferable evidence. GRPO gains at 45M parameters are real but conditional: +6 to +30 points where pretraining or SFT gave partial competence, and zero where it gave none. MicroReason's depth curve (non-zero accuracy at every depth from 1 to 5) is the favorable case.
- The TinyZero failure at 0.5B came from a pretrained general LM doing open-ended search. It probably does not transfer to a task-specialized 43M model whose SFT already reaches 37–95%.

### Gaps
- No Mini-R1 numbers were retrieved. No paper doing GRPO on toy transformers for multi-digit addition with length generalization was found.

## Does RL on harder compositions improve depth generalization?

### Takeaway
Yes, if RL is trained at or near the depth frontier and the atomic skills already exist. In the f(g(x)) paper (Llama-3.1-8B), RL on depth-2 compositions raised depth-3 accuracy from about 5% to about 30% and depth-4 from about 1% to about 15%. RL on depth-1 alone did nothing for depth 3 and above. The 45M study agrees that RL at depth 2 helps depth 3 and RL at depth 1 does not, but finds no extrapolation to depths never seen in pretraining.

### Cited Findings
- **"From f(x) and g(x) to f(g(x))" (Yuan et al., arXiv 2509.25123) [LLM-only, Llama-3.1-8B-Instruct after a stage-1 RFT base; string-transformation functions].**
  - RL on Level 1 only peaked at about 90% on Level 1, stayed below 25% on Level 2, and was near zero at Level 3 and above.
  - RL on Level 2 reached 64% on Level 2, about 27–30% on Level 3 and about 15% on Level 4, and kept improving at Level 5. The RFT base was about 5% on Level 3 and 1% on Level 4.
  - Iterative RFT on the same Level-2 data reached only 15% on Level 2 and never exceeded 2.6% on Level 3.
  - Pass@k: the gap between RL and base widens at harder levels, from 4% at pass@1 to about 25% at pass@1000/1024 on Level 5.
  - Countdown transfer: the Multi-Base model reached about 17% on Level 3. Adding RL on Level 1 gave about 20%, and RL on Levels 1+2 gave about 35% on Level 3 and 6% on Level 4. Transfer failed completely without the target task's atomic skills.
  - A blog review notes that gains fade at Level 3 and above.
  - Source: [arXiv HTML](https://arxiv.org/html/2509.25123v3); [abstract](https://arxiv.org/abs/2509.25123v3); [blog review](https://leonericsson.github.io/blog/2025-09-02-fromfandg)
- **45M from-scratch study:** GRPO does not extrapolate to unseen depths, GRPO at 2 steps helps 3-step tasks, and GRPO at 1 step does not. — [arXiv 2510.15990](https://arxiv.org/html/2510.15990v1)

### Inferences
- For MicroReason, RL prompts should emphasize compositional depth of 2 or more operators (3–5 leaves), where accuracy sits between 37% and 80%, rather than depth-1 prompts, where it is 95%. Expect transfer to one or two depths beyond the RL training distribution only if SFT gave non-zero accuracy there. Generating SFT data with 7+ leaves could test this.
- Relative gains are likely largest at the depth frontier: in the f(g(x)) paper, depth 3 went from about 5% to 30%.

### Gaps
- No study measured gains on arithmetic expression trees specifically. The f(g(x)) result is on an 8B model.

## Rollout temperature and diversity tricks to raise mixed-group rates

### Takeaway
The best-supported temperatures are 1.0 for general use (Skywork-OR1) and, in POLARIS, the highest temperature before quality collapses (1.4 for Qwen3-4B), raised at each stage as entropy falls. Temperature 0.6 collapsed entropy within about 100 steps in Skywork-OR1. No source studied temperature on models under 100M parameters.

### Cited Findings
- **Skywork-OR1:** at τ=0.6, entropy started extremely low or collapsed to near zero within about 100 steps. τ=1.0 gave greater later-stage gains. Limiting early entropy collapse mattered for test performance. — [Skywork-OR1](https://arxiv.org/html/2505.22312v2); [abstract](https://arxiv.org/abs/2505.22312v1)
- **POLARIS:**
  - Qwen3-4B zones: robust at about 0.6–1.4, controlled exploration at 1.4–1.55, collapse at about 1.55 and above. The starting temperature was 1.4 for Qwen3-4B and 0.7 for R1-Distill-7B.
  - Temperatures per stage were 1.4, 1.45 and 1.5 (4B) and 0.7, 1.0 and 1.1 (7B), stepping by +0.05 when entropy drops slightly.
  - The common settings of 0.6 and 1.0 degraded performance for these models.
  - Source: [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris)

### Inferences
- MicroReason's temperature 1.0 gave only 19.5% mixed groups, and the notes in memory show the RL gate failed at both earlier sampling settings (0.182 and 0.195 mixed against a 0.20 threshold). Changing the prompt distribution through filtering probably matters more than changing temperature. A POLARIS-style sweep (1.0–1.5, checking where answer accuracy collapses) is cheap at 43M parameters, but prompt filtering directly sets the mixed rate to about 100%.
- The all-or-nothing outcome pattern suggests per-prompt determinism: the SFT model is confident on each prompt, so moderate temperature changes move few prompts into the mixed band.

### Gaps
- No small-model temperature study was found. Other diversity tricks (top-p, prompt paraphrase or variant prompts) were not covered by retrieved sources.
