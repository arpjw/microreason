# Reward Design Beyond 0/1 Outcome Reward, and Statistical Design for RL Claims (43M from-scratch arithmetic model)

Scope note: nearly all of the evidence below comes from large pretrained LLMs (1.1B to 70B+, mostly 7B Qwen/Mistral/DeepSeek models on GSM8K/MATH/AIME). I found no paper that runs process or step rewards on a tiny transformer trained from scratch on synthetic arithmetic. Each finding is tagged [LLM-only] where that applies.

## 1. Process vs outcome supervision: what gains are measured over outcome-only?

### Takeaway
Process or dense rewards give a clear, measured benefit mainly in two places: reranking/search (PRM beats ORM) and RL sample efficiency (5-6x for progress-based rewards). Gains in final accuracy over a well-tuned outcome-only RL baseline are smaller, around +6 points at best, and sometimes zero or negative when the dense reward is learned. All of this evidence is [LLM-only]. The cleanest signal is that process supervision mostly improves the correctness of the reasoning trace, not final-answer accuracy.

### Cited Findings
- Uesato et al. 2022 (DeepMind, GSM8K): final-answer error went from 16.8% (previous best) to 12.7%, and reasoning (trace) error among final-correct solutions went from 14.0% to 3.4%. Pure outcome-based supervision gave *similar final-answer error* with less label supervision. Process-based supervision, or a learned RM that emulates it, was needed for correct reasoning steps. [LLM-only; the paper's base models are large Chinchilla-family LMs, and the model is not named on the abstract page] — [arXiv 2211.14275](https://arxiv.org/abs/2211.14275)
- Lightman et al. 2023 ("Let's Verify Step by Step"): a process-supervised RM solves 78% of a representative MATH test subset (best-of-N reranking, GPT-4-based). PRM800K has 800,000 human step-level labels. "Process supervision significantly outperforms outcome supervision," and active learning markedly improves it. This is a reranking result, not an RL result. [LLM-only] — [arXiv 2305.20050](https://arxiv.org/abs/2305.20050)
- Math-Shepherd (automatic step labels from rollouts): step-by-step PPO with Math-Shepherd took Mistral-7B from 77.9% to 84.1% on GSM8K and from 28.6% to 33.0% on MATH. As a verifier for reranking it reached 89.1% / 43.5%. [LLM-only, 7B] — [arXiv 2312.08935](https://arxiv.org/abs/2312.08935)
- PRIME (implicit process rewards, starting from Qwen2.5-Math-7B-Base): Eurus-2-7B-PRIME gets a 15.1% average improvement over its SFT model and surpasses Qwen2.5-Math-7B-Instruct on seven benchmarks with 10% of its training data. The abstract gives no head-to-head number against an outcome-only RL baseline. [LLM-only, 7B] — [arXiv 2502.01456](https://arxiv.org/abs/2502.01456)
- Setlur et al. 2024 ("Rewarding Progress", process advantage verifiers): progress is defined as the change in the probability of eventually being correct, measured under a separate prover policy. PAVs are >8% more accurate than ORMs in test-time search and 1.5-5x more compute-efficient. In online RL, dense PAV rewards give a 5-6x sample-efficiency gain and a >6% accuracy gain over ORMs. [LLM-only; I could not confirm model sizes from the abstract, see Gaps] — [arXiv 2410.08146](https://arxiv.org/abs/2410.08146)
- VinePPO (Monte Carlo value estimates at each step for step-level credit assignment, replacing PPO's value network): it beats PPO on MATH and GSM8K, reaching higher test accuracy for a given training accuracy, in up to 3.0x less wall-clock time. [LLM-only] — [arXiv 2410.01679](https://arxiv.org/abs/2410.01679)
- Gao et al. 2024 ("On Designing Effective RL Reward at Training Time for LLM Reasoning", 1.5B and 7B models, MATH/GSM8K): learned ORM/PRM rewards combined with the success reward "may NOT help or even hurt RL training" compared with the success reward alone. — [arXiv 2410.15115](https://arxiv.org/abs/2410.15115)

### Inferences
- In the papers that run RL, the gain from process rewards shows up mostly as sample efficiency (Setlur 5-6x, VinePPO 3x wall-clock). The final-accuracy gain over outcome-only RL is about +6 points at best (Setlur), and Gao et al. found no gain or a loss. Expecting a large greedy-accuracy jump on held-out shapes from step rewards alone is not supported by the literature.
- The arithmetic setting has a key advantage the LLM papers lack. Step correctness is checkable *exactly* by a program, so the step reward is not a learned PRM and cannot be adversarially exploited the way learned PRMs are. That said, the exact check itself can still be gamed (see section 2).
- Most relevant to this project: protocol v3 Phase C stopped twice at the reward-variance gate (mixed-outcome groups 0.182 and then 0.195 < 0.20). A dense per-step reward creates within-group variance even when every rollout in a group gets the final answer wrong. That directly addresses the zero-advantage problem that DAPO's dynamic sampling filters out (section 4). On deep expressions (~37% at 5 operators), groups where all rollouts fail differ in how far the trace gets before the first error.
- The Setlur "progress" framing suggests a cheap exact analogue. Reward a step by the change in Monte Carlo success probability from that state, or simply by whether the reduced sub-expression is exactly correct. VinePPO's MC value estimates are also cheap for a 43M model, because rollouts from intermediate states cost little.

### Gaps
- No paper found that measures process vs outcome RL on a small (<100M) transformer trained from scratch on synthetic arithmetic.
- I could not verify the exact model sizes for Setlur et al. (I recall the Gemma 2B/9B/27B family) or VinePPO (I recall RhoMath-1.1B and DeepSeekMath-7B), or VinePPO's gradient-step savings figure. Treat these as unverified.
- PRIME's sample-efficiency number against outcome-only RL (I recall about 2.5x in the paper body) is not on the abstract page and is unverified.

## 2. Rule-based step-verification rewards (prefix of correct steps, first-error position) and shaping pitfalls

### Takeaway
Deterministic, rule-based step verifiers are an emerging line of work and look promising: +6.5 F1 over verifiable outcome rewards in one 2026 study, though outside math. The documented failure mode of step-count rewards is that the policy inflates the number of "correct" steps, for example by repeating correct but unnecessary steps. The standard defence is to cap or difference the step reward (Gao et al.'s Clip and Delta), so that total step credit is bounded and cannot outgrow the outcome reward.

### Cited Findings
- Gao et al. 2024: a policy can earn high reward by "repeating correct but unnecessary reasoning steps." Their two fixes, Clip and Delta, cap the accumulated reward of any trajectory, keeping the dense reward useful without letting it be exploited. With these, RL improved all evaluated 1.5B/7B LLMs, including Qwen2.5-Math-7B-Instruct on MATH and GSM8K. [LLM-only] — [arXiv 2410.15115](https://arxiv.org/abs/2410.15115)
- VSRM (verifiable stepwise reward mechanism): a rule-based reward for intermediate states, with no trained PRM. Stepwise rewards are assigned at truncation points from the change in correctness of sub-rollouts. This is an MC-progress style reward computed from verifiable outcomes. [LLM-only] — [arXiv 2508.10293](https://arxiv.org/abs/2508.10293v2)
- Verifiable Process Reward Models (VPRMs, Findings of ACL 2026): each intermediate step is checked by deterministic, rule-based verifiers. Results are up to 20% higher F1 than SOTA and 6.5% higher than verifiable outcome rewards, with "substantially higher coherence" between steps and final labels. The domain is medical risk-of-bias assessment, not arithmetic. Reward hacking is cited as a weakness of neural judges that motivates rule-based checking. — [arXiv 2601.17223](https://arxiv.org/abs/2601.17223)
- A Stanford CS224R project on Countdown arithmetic puzzles with Qwen2.5-0.5B (VCC-RLOO) scans rollouts for the first verified answer and adds a contrastive loss. It is the closest small-model arithmetic RL result I found, but it uses a pretrained model and is a course report, not peer-reviewed. — [CS224R report](https://cs224r.stanford.edu/projects/pdfs/Zayn%20Malhotra%20Ziyi%20Ding%20submission_416287508/CS224R_REPORT_V3.pdf)

### Inferences
These are design implications for the 43M model and are not tested in any source.
- **Hacking surfaces for an exact step checker on expression-tree reductions:**
  - (a) Extra trivially correct steps, such as re-copying an unchanged expression or splitting one reduction into many no-op lines. This is the Gao et al. "repeat correct steps" pattern.
  - (b) If the reward is "number of correct steps," *shorter* traces earn less, so the policy is pushed toward padding. If the reward is "fraction of steps correct," shorter traces (fewer chances to err) or skipping hard reductions are favored. A 2-step trace that jumps to a wrong answer could score 1/1 correct steps before the jump.
  - (c) Steps that are locally valid but don't make progress, such as applying the right operation to the wrong sub-tree.
- **Safer exact formulations:**
  - (i) The *correct-prefix fraction relative to the gold number of reductions*: credit = (number of consecutive correct, progress-making reductions before the first error) / (number of reductions the canonical solution needs). This is bounded in [0,1], cannot be inflated by padding, and does not reward short traces.
  - (ii) Additionally require each counted step to strictly reduce tree size (operator count decreases by exactly one). This kills no-op padding.
  - (iii) Give zero step credit after the first error, so "first-error position" is the signal.
  - (iv) Cap total step credit below the outcome reward, as in Gao et al.'s Clip. For example, outcome 1.0 + step credit ≤ 0.5 total, so that a fully correct trajectory always beats any wrong one.
- For the linear-equation traces (Expand/Combine/solve), the same rules apply. Each canonical phase is either verified-correct or not, and the gold phase count is known, so prefix-fraction credit is well defined.

### Gaps
- No source directly studies "correct-prefix" or "first-error-position" rewards in synthetic math RL. The formulations above are design inferences.
- No numbers found on how often step-count hacking appears at small scale.

## 3. Partial credit for numeric closeness: why it is usually avoided

### Takeaway
I found no primary source that systematically evaluates numeric-closeness rewards for math RL. The mainstream RLVR recipes use binary exact-match plus format checks. The arguments against numeric closeness are inferential: it rewards off-by-a-bit guesses rather than a correct procedure, and in exact arithmetic one wrong step often gives a numerically "close" answer only by accident.

### Cited Findings
- DeepSeek-R1-style rule-based RLVR uses final-answer correctness plus a format check (reasoning wrapped in think tags), with no closeness credit. This comes from a secondary glossary summary and should be checked against the R1 paper. — [howaiworks.ai RLVR glossary](https://howaiworks.ai/glossary/reinforcement-learning-from-verifiable-rewards)
- Gao et al.'s broader finding: any auxiliary reward that does not track correctness tends to be exploited, so total auxiliary credit must be bounded. — [arXiv 2410.15115](https://arxiv.org/abs/2410.15115)

### Inferences
- For multi-step integer arithmetic, |pred − gold| has a bad geometry. A single sign error or wrong operator early in the tree can produce a huge numeric error, while a sloppy guess can land close. Closeness is therefore poorly correlated with "how much of the procedure was right." Step-level exact verification measures that directly and dominates numeric closeness here.
- Numeric closeness also invites a "guess a plausible magnitude" strategy that does not transfer to held-out shapes. Recommendation: do not use it. Use exact step-prefix credit instead.

### Gaps
- No primary paper found that quantifies numeric-closeness rewards vs binary rewards in math RL.

## 4. Combining outcome + step reward (weights); format, length and overlong penalties

### Takeaway
No source gives a validated weight like "outcome 1.0 + 0.1 per correct step" for small models. The strongest guidance is structural. Bound the dense term (Clip/Delta), make sure a fully correct answer always dominates, and remove or penalize truncation-driven length artifacts. DAPO's ablation is the best-quantified evidence that length handling and group-variance handling matter: on Qwen2.5-32B, AIME24 went from 30 (naive GRPO) to 50 (DAPO) [LLM-only].

### Cited Findings
- DAPO soft overlong punishment: reward 0 if |y| ≤ L_max − L_cache, linear penalty ((L_max − L_cache) − |y|)/L_cache in the buffer zone, and −1 beyond L_max. They used L_max = 16,384 and L_cache = 4,096 (penalty starts at 12,288 tokens; generation cap 20,480). Overlong *filtering* masks the loss of truncated samples. — [DAPO, arXiv 2503.14476](https://arxiv.org/html/2503.14476v2)
- DAPO ablation on AIME24 avg@32, Qwen2.5-32B base: naive GRPO 30, + overlong filtering 36, + clip-higher 38, + soft overlong punishment 41, + token-level loss 42, + dynamic sampling 50. The reference point is DeepSeek-R1-Zero-Qwen-32B at 47. [LLM-only, 32B] — [arXiv 2503.14476](https://arxiv.org/html/2503.14476v2)
- DAPO's token-level loss motivation: sample-level averaging under-weights tokens in long responses and fails to penalize gibberish or repetition there, which leads to unhealthy growth in entropy and length. — [arXiv 2503.14476](https://arxiv.org/html/2503.14476v2)
- DAPO dynamic sampling: drop prompts whose group accuracy is exactly 0 or 1 (zero advantage) and keep sampling until the batch is full of mixed-outcome prompts. Total training time was not significantly affected, and convergence came sooner. — [arXiv 2503.14476](https://arxiv.org/html/2503.14476v2)
- Gao et al.: Clip/Delta bound the accumulated dense reward so it stays useful without being exploited. — [arXiv 2410.15115](https://arxiv.org/abs/2410.15115)

### Inferences
These apply to the 43M project.
- **Weighting:** use r = 1.0·[final exact] + λ·(correct-prefix fraction), with λ in roughly 0.1-0.3 and the prefix term in [0,1]. This guarantees that any correct-final trajectory (≥1.0) beats any incorrect one (≤λ<1). A per-step additive bonus (+0.1 per correct step) is unbounded in trace length, invites padding, and should be avoided unless clipped. Pre-register λ and a small ablation (λ=0 control vs one λ value), not a sweep.
- **Under GRPO's group normalization, the absolute scale of λ matters less than you might expect.** In a group where every rollout fails the final answer, the advantage comes entirely from prefix differences, however small λ is. Once the std normalization amplifies them, they are as strong as outcome differences. This is the intended fix for the mixed-group-fraction gate failure, and also its risk: in all-fail groups the policy is fully optimizing the step proxy. A safeguard is to anneal λ, or to monitor whether final accuracy on deep expressions rises along with prefix credit.
- **Length/format:** traces here are short and canonical, so DAPO-scale overlong shaping is mostly unnecessary. Still, (a) hard-cap length at about 1.5-2x the longest gold trace, and (b) give reward 0 and mask or penalize any trace that fails to parse. This prevents the "never finish" strategy that a step-prefix reward could otherwise tolerate. Pre-register a guard metric: mean trace length and the number of reductions per trace vs gold, with a stop rule if they drift.
- **Dynamic sampling** (filter all-0/all-1 groups) is the outcome-only alternative to dense rewards for the same variance problem. It costs more samples per batch when the mixed-group fraction is about 0.19.

### Gaps
- No source validates specific outcome+step weights for small models, or reports whether dense rewards help or hurt more at small scale.
- I did not verify Kimi k1.5's length penalty or Dr. GRPO's length-bias analysis (arXiv 2503.20783) in this session.

## 5. Statistical practice: seeds, variance, CIs, power; is ≥2 points realistic with ~6,000 test prompts?

### Takeaway
With ~6,000 prompts and a *paired*, per-prompt comparison of greedy outputs, evaluation-sampling noise is small: SE of the difference ≈ 0.4-0.6 points. A true gain of ≥2 points is detectable on a single seed pair. The dominant uncertainty is instead (a) *training-seed* variance of the RL run, which the prompt-level bootstrap does not capture, and (b) clustering by expression shape, which can inflate SEs up to about 3x. Note also that a rule requiring the *point estimate* ≥2 has only ~50% power when the true gain is exactly 2.

### Cited Findings
- Hochlehnert et al. 2025 ("A Sober Look"): results are highly sensitive to seeds, decoding, prompt format and hardware. Across-seed SD on AIME'24 is about 4-8 points for 1.5B/7B R1-distill models (R1-Distill-1.5B 28.7 ± 4.8; Open-RS2 31.3 ± 7.7). On MATH500 (500 problems) it is only about 0.2-1.9 points. Pass@1 SDs of roughly 5-15 points appear on small benchmarks, and one AIME'24 question moves Pass@1 by about 3.3 points. They recommend ≥30 seeds for small benchmarks (they used 10 seeds for AIME/AMC and 3 elsewhere), with paired one-sided t-tests / Wilcoxon at p<0.01. [LLM-only] — [arXiv 2504.07086](https://arxiv.org/html/2504.07086v2)
- Re-evaluated RL gains for 1.5B models vs R1-Distill: Open-RS1/2/3 were all n.s. (for example, MATH500 +0.2/−0.8/−0.7), whereas the original claims were 10-15% gains. STILL-3 was n.s. everywhere. DeepScaleR: +2.9 on MATH500 (significant), but +8.3 on AIME'24 was n.s. Most RL methods show "only modest improvements," SFT generalized more consistently, and RL appears prone to overfitting small benchmarks. — [arXiv 2504.07086](https://arxiv.org/html/2504.07086v2)
- Chandak, Goel and Prabhu 2025: reviewed seven popular LLM-RL papers. Pre-RL baselines were often underreported compared with official numbers, and in several cases the post-RL model was *worse* than a correctly evaluated baseline. — [Notion write-up](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37); [LessWrong mirror](https://www.greaterwrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evaluations-call-into-question-recent-llm)
- Miller 2024 ("Adding Error Bars to Evals"):
  - Recommendations: CLT standard errors; *clustered* SEs when questions come in related groups (up to about 3x larger than naive in real examples); resample answers or use next-token probabilities to cut variance; inference on *question-level paired differences*; power analysis before running.
  - Paired SE: sqrt(SE_A² + SE_B² − 2·SE_A·SE_B·Corr).
  - MDE = (z_{α/2}+z_β)·sqrt((ω² + σ_A²/K_A + σ_B²/K_B)/n). Worked example: δ=3 points, α=0.05, 80% power, ω²=1/9 needs n≈969.
  - Do not pool SEs over K·N resamples, and do not lower temperature to reduce variance.
  — [arXiv 2411.00640](https://arxiv.org/html/2411.00640v1)
- Agarwal et al. 2021 ("Deep RL at the Edge of the Statistical Precipice"): report interval estimates of aggregate performance rather than point estimates, use performance profiles, and use robust aggregates such as the interquartile mean. In their Atari 100k case study, point-estimate conclusions differed substantially from the fuller analysis. (Their stratified-bootstrap CI method and the rliable library are well known but were not confirmed on the abstract page.) — [arXiv 2108.13264](https://arxiv.org/abs/2108.13264)

### Inferences (power calculations for this project; arithmetic is mine, formulas from Miller)
- **Paired SE with greedy decoding (σ²=0, a deterministic per-prompt score):** for paired 0/1 outcomes the variance of the per-prompt difference ≈ d − Δ², where d is the discordant-pair rate (prompts where exactly one model is right). With n=6,000:
  - d=0.10 → SE≈0.41 pts, 95% CI ±0.80, MDE(80% power)≈1.15 pts
  - d=0.20 → SE≈0.58 pts, ±1.13, MDE≈1.62 pts
  - d=0.30 → SE≈0.71 pts, ±1.39, MDE≈1.98 pts
  - An *unpaired* comparison at p≈0.6 gives SE≈0.89 pts and MDE≈2.5 pts. Pairing is essential.
- **Threshold-rule caveat:** if the pre-registered success rule is "point estimate ≥ +2.0 AND 95% CI lower bound > 0," then a true gain of exactly 2.0 passes only about 50% of the time. To reach ~80% power on the "≥2" part, the true gain needs to be about 2 + 0.84·SE ≈ 2.4-2.6 points. If the scientific claim is "gain ≥ 2," the cleaner rule is "CI lower bound ≥ 0" plus a separately reported effect size, or a one-sided test against 2 points (which needs a true gain of ~3+).
- **Cluster by held-out shape:** test prompts sharing an expression shape or template are correlated. If held-out shapes are few (tens), use a clustered SE or a cluster bootstrap that resamples shapes, then prompts within shapes. Miller's up-to-3x inflation would turn a 0.5-point SE into 1.5 points, making a 2-point MDE unattainable. Pre-register the number of distinct held-out shapes as a power input.
- **Seeds:** the prompt-level paired bootstrap answers "is this checkpoint better than SFT on this test set," not "does this RL recipe reliably help." For the recipe claim:
  - Use ≥3, preferably 5, RL training seeds. Report each seed's paired Δ with its CI, plus the mean Δ across seeds with a CI from a hierarchical bootstrap (resample seeds, then prompts) or a t-interval over seed means.
  - Pre-register that *every* seed (or ≥4/5) must clear CI-lower>0. This guards against the single-lucky-seed failure that Hochlehnert et al. document.
  - With 2 seeds the across-seed SD is essentially unestimated.
- **Baselines (Chandak et al.):** evaluate the SFT baseline with the identical greedy decoder, parser, prompt and checkpoint-selection rule. Include an "extra-SFT-compute" control matched in GPU-hours, so the RL gain isn't just more training.
- **Checkpoint selection:** choose the RL checkpoint on held-out-val only, by a pre-registered rule (for example, the last checkpoint, or best val), and look at test once. Otherwise selection inflates the gain.
- Because decoding is greedy, the resampling (K) advice in Miller applies only if pass@k or sampled metrics are also reported. Under greedy decoding, only ω² (prompt heterogeneity) and training-seed variance matter.

### Gaps
- I found no published across-seed variance numbers for RL on small from-scratch transformers. The project should estimate seed SD from its own runs; pilot runs (e.g., the GRPO v2 seeds) can supply it.
- I did not verify Henderson et al. 2018 ("Deep RL that Matters") or Colas et al. 2018 ("How many random seeds?") in this session.
- I did not verify whether "Spurious Rewards" (2025) results, where Qwen models improve under random rewards, extend beyond Qwen.
