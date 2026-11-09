# GRPO-family algorithm variants and hyperparameters for outcome-reward RL on a ~43M from-scratch transformer

Scope note: almost all evidence below comes from models of 0.5B to 32B parameters, mostly in the Qwen2.5/Qwen3 families. The only RL-from-scratch evidence near our scale is the CMU 100M-parameter synthetic study (arXiv 2512.07783). Every number is tagged with the model size it was measured at.

## DAPO: clip-higher, dynamic sampling, token-level loss, overlong shaping

### Takeaway
DAPO sets eps_low=0.2 and eps_high=0.28, uses no KL, LR 1e-6, 512 prompts x 16 samples, and 16 gradient updates per rollout. Dynamic sampling was the largest single ablation step (+8 AIME points). All of this was measured on a 32B base model with 16k-token rollouts, so the overlong and token-level components matter far less for our 76-token traces.

### Cited Findings
- Model: Qwen2.5-32B base. eps_low = 0.2, eps_high = 0.28. KL term removed entirely. — [DAPO, arXiv 2503.14476](https://arxiv.org/html/2503.14476)
- Optimizer AdamW, constant LR 1e-6, linear warmup over 20 rollout steps. — [DAPO](https://arxiv.org/html/2503.14476)
- Prompt batch 512, group size 16 responses per prompt, mini-batch 512, giving 16 gradient updates per rollout step. The data are therefore partly off-policy. — [DAPO](https://arxiv.org/html/2503.14476)
- Lengths: 16,384 expected max plus a 4,096-token soft-punish buffer, for 20,480 maximum tokens. The penalty grows linearly inside the buffer and reaches -1 beyond the max. "Overlong filtering" instead masks the loss of truncated samples. — [DAPO](https://arxiv.org/html/2503.14476)
- Ablation on AIME24 (avg@32, 32B): naive GRPO 30; + overlong filtering 36; + clip-higher 38; + soft overlong punishment 41; + token-level loss 42; + dynamic sampling (full DAPO) 50. The DeepSeek-R1-Zero-Qwen-32B reference scored 47. DAPO reached it in about 50% of that run's steps. — [DAPO](https://arxiv.org/html/2503.14476)
- Naive GRPO/PPO shows rapid entropy collapse, with responses inside a group becoming nearly identical. Clip-higher counters this. Token-level loss stops long gibberish or repetitive samples from inflating entropy and length. A slow upward entropy trend went with better performance. — [DAPO](https://arxiv.org/html/2503.14476)
- Dynamic sampling filters out groups that are all correct or all wrong, and oversamples until the batch is full of mixed groups. — [DAPO](https://arxiv.org/html/2503.14476)
- Overlong filtering helps substantially at an 8k threshold, and its benefit "diminish[es] significantly" at 20k (Qwen3-4B/8B). — [Tricks or Traps, arXiv 2508.08221](https://arxiv.org/html/2508.08221)

### Inferences
- Our traces are about 76 tokens with a 256-token cap, so truncation should be rare. Overlong filtering and soft punishment are near-free safety nets: mask truncated samples, and skip the length penalty. Token-level versus sample-level aggregation differs little when lengths vary little. That component was worth about +1 point even at 32B.
- Dynamic sampling, the biggest DAPO gain, was already in our failed run. DAPO alone therefore will not close our gap. What remains is clip-higher, plus making more prompts informative through data selection and larger groups (see the hyperparameter section).
- The DAPO setup of 16 updates per rollout batch conflicts with Skywork-OR1's finding that off-policy updates speed up entropy collapse (see the staleness section). For a tiny model I would choose on-policy updates (1 update per batch) over DAPO's 16.

### Gaps
- DAPO does not state the training sampling temperature in the parts I extracted. Its evaluation used temperature 1.0 and top-p 0.7.
- DAPO reports no ablation below 32B.

## Dr. GRPO: length-normalization and std-normalization biases

### Takeaway
Dr. GRPO removes both the per-response 1/|o_i| term and the division by the group reward std, and normalizes by a constant such as the generation budget. This stops length inflation of wrong answers and gave better token efficiency and accuracy across three seeds at 1.5B to 7B. The std term in particular overweights prompts with near-uniform rewards, which is exactly our regime of pass rates near 0 or 1.

### Cited Findings
- Length bias: dividing by |o_i| gives short correct responses larger updates and penalizes long incorrect responses less, so the policy drifts toward long wrong answers. — [Dr. GRPO, arXiv 2503.20783](https://arxiv.org/html/2503.20783)
- Difficulty bias: dividing by the group reward std gives larger weight to questions whose rewards are nearly uniform (almost all right or almost all wrong). — [Dr. GRPO](https://arxiv.org/html/2503.20783)
- Fix: drop 1/|o_i| and the std normalization, and replace the per-response mask sum with a constant (for example the generation budget). The loss then matches the unbiased PPO objective. — [Dr. GRPO](https://arxiv.org/html/2503.20783)
- Effect: response length stops growing rapidly, and incorrect responses become substantially shorter. Both "remove length norm" and full Dr. GRPO beat vanilla GRPO on training reward and evaluation accuracy. Improvements in token efficiency and final accuracy were statistically significant across 3 seeds. — [Dr. GRPO](https://arxiv.org/html/2503.20783)
- Oat-Zero hyperparameters (Qwen2.5 1.5B/3B/7B): LR 1e-6 constant; group size 8; temperature 1.0 (top-p 1.0); max response 3,000 tokens; KL coefficient 0.0; AdamW (0.9, 0.95), no weight decay; gradient clip 1.0; 1 inner update epoch; clip eps 0.2; 8xA100 for about 1 day. — [Dr. GRPO](https://arxiv.org/html/2503.20783)
- Minimalist recipe (Qwen2.5-Math-7B, MATH levels 3–5): 43.3% AIME24, 51.4% average. Llama-3.2-3B gains are small (3.3 to 6.8 average) unless the model was math-pretrained first (FineMath 14.8, NuminaQA 20.7). The authors argue that math pretraining raises the RL ceiling. — [Dr. GRPO](https://arxiv.org/html/2503.20783)
- "Tricks or Traps" (Qwen3-4B/8B): removing std helps when rewards are highly concentrated (easy data), because it prevents gradient amplification from a small denominator. Group-level mean with batch-level std was the most robust choice. — [Tricks or Traps](https://arxiv.org/html/2508.08221)
- ScaleRL (8B): prompt-level, batch-level and no std normalization performed similarly. Batch-level was adopted as marginally better. — [ScaleRL, arXiv 2510.13786](https://arxiv.org/html/2510.13786)
- TRL exposes these options as `loss_type="dr_grpo"` and `scale_rewards` in {"group", "batch", False}. Its defaults are `loss_type="dapo"` and `scale_rewards="group"`. — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)

### Inferences
- With 0/1 rewards and group size G=8, a group with 1 correct out of 8 has std of about 0.33. Group std normalization then scales its advantages by about 3x relative to a balanced 4/8 group. Our bimodal pass-rate distribution makes us the "concentrated rewards" case where removing std (or using batch std) is recommended. Pre-registering `scale_rewards=batch` or none, with a constant-denominator loss (Dr. GRPO), is well supported.
- Length bias matters little at roughly 76-token traces, but the constant-denominator loss costs nothing.

### Gaps
- Dr. GRPO does not report batch size or step count in the extracted text.
- I found no Dr. GRPO result below 1.5B.

## Other variants: GSPO, CISPO, REINFORCE++, RLOO, VAPO, PPO with a value head

### Takeaway
GSPO (sequence-level ratios) and CISPO (clipped importance-sampling weights) improve stability and asymptotic performance, but the evidence comes from 8B to 30B MoE models where token-level ratio noise and expert routing are the problems. Batch-normalized baselines (REINFORCE++, Lite PPO) are argued to generalize better than per-group baselines. I found no head-to-head comparison of these methods at sub-1B scale.

### Cited Findings
- GSPO: the importance ratio is s_i = (pi_theta(y_i|x) / pi_old(y_i|x))^(1/|y_i|), a geometric mean of the token ratios, with sequence-level clipping. Clip range is left 3e-4, right 4e-4, versus GRPO's 0.2 / 0.27. GSPO clips about two orders of magnitude more tokens yet trains more efficiently. GRPO showed instability and sometimes irreversible collapse on Qwen3-30B-A3B, and needed "Routing Replay" to converge. Each rollout batch was split into 4 mini-batches. — [GSPO, arXiv 2507.18071](https://arxiv.org/html/2507.18071)
- ScaleRL (8B dense and 17Bx16 MoE): both GSPO and CISPO raised the asymptotic pass rate over DAPO, and CISPO was chosen as marginally better late in training. Swapping CISPO back to DAPO within ScaleRL gave a similar asymptote but lower efficiency (B = 1.77 vs 2.01). — [ScaleRL](https://arxiv.org/html/2510.13786)
- ScaleRL final recipe: PipelineRL (8-step off-policyness), forced length interruptions, FP32 logits at the LM head, CISPO, prompt-level loss averaging, batch-level advantage normalization, zero-variance filtering, and "No-Positive-Resampling" (permanently drop prompts with pass rate at or above 0.9). — [ScaleRL](https://arxiv.org/html/2510.13786)
- In TRL, `loss_type="cispo"` defaults `epsilon_high` to 5.0, following ScaleRL. `importance_sampling_level="sequence"` gives GSPO. — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- REINFORCE++ uses global batch advantage normalization with no critic. The "with baseline" variant subtracts the group mean and then normalizes globally. The authors argue that per-prompt baselines (GRPO, RLOO) overfit. In their example, GRPO trained on 30 AIME-24 problems reached about 95 Pass@1 on train and 0.0 on test. These are the authors' own experiments, not independent replications. — [REINFORCE++, arXiv 2501.03262](https://arxiv.org/html/2501.03262v8)
- Lite PPO (group mean + batch std, token-level loss, vanilla PPO loss with no critic) outperformed GRPO and DAPO on Qwen3-4B-Base and 8B-Base. No numeric gain is stated in the text. — [Tricks or Traps](https://arxiv.org/html/2508.08221)
- Token-level aggregation was better for base models, and sequence-level aggregation was better for aligned (instruction-tuned) models in most settings (Qwen3-4B/8B). — [Tricks or Traps](https://arxiv.org/html/2508.08221)
- ScaleRL found prompt-level averaging gave the highest asymptote among sample, prompt and token averaging (8B). — [ScaleRL](https://arxiv.org/html/2510.13786)

### Inferences
- GSPO's motivation (MoE routing drift, long-sequence token-ratio noise) mostly does not apply to a 43M dense model with 76-token outputs. If updates are fully on-policy (1 update per batch), all importance ratios equal 1, and GRPO, GSPO, CISPO and DAPO collapse to the same REINFORCE-with-baseline gradient. The algorithm choice then reduces to advantage estimation (baseline and normalization) and loss aggregation.
- Our model is SFT'd, i.e. "aligned" in the Tricks-or-Traps sense, which weakly favors sequence-level aggregation over token-level. Evidence is from 4B/8B only.
- RLOO's leave-one-out mean is an unbiased baseline. With G=8 and 0/1 rewards it differs from the GRPO mean only by a factor of G/(G-1). It is a reasonable no-std default.

### Gaps
- I did not fetch the original CISPO paper (MiniMax-M1), so the exact CISPO formula and hyperparameters come only via ScaleRL and TRL.
- I did not fetch VAPO or Open-Reasoner-Zero (PPO with a value head), so I have no verified numbers for value-model methods. A critic trained from scratch on a 43M model with 0/1 terminal reward is plausible but unevidenced.
- I found no independent comparison of RLOO, REINFORCE++ and GRPO at sub-1B scale.

## KL penalty vs no KL, and reference resets

### Takeaway
Most 2025 recipes that start from base models use KL=0 (DAPO, Dr. GRPO, Skywork-OR1, POLARIS, Tricks or Traps, TRL default). Recipes that start from an SFT'd or distilled checkpoint often keep a small KL: 1e-3 at 100M (CMU) and 1e-4 at 0.5B–14B (SimpleRL-Zoo). ProRL keeps KL and hard-resets the reference policy and optimizer when validation plateaus. Skywork found that KL=1e-3 stalled gains in multi-stage training.

### Cited Findings
- DeepSeekMath GRPO (7B): KL coefficient 0.04. — [DeepSeekMath, arXiv 2402.03300](https://arxiv.org/html/2402.03300)
- DeepSeek-R1 used KL 0.001 (as cited in TRL docs). TRL defaults to beta = 0.0 and cites Open-Reasoner-Zero, Dr. GRPO and DAPO as showing KL is not essential. — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- DAPO (32B): no KL. Dr. GRPO (1.5B–7B): KL 0.0. Tricks or Traps (4B/8B): init_kl_coef 0.0. — [DAPO](https://arxiv.org/html/2503.14476); [Dr. GRPO](https://arxiv.org/html/2503.20783); [Tricks or Traps](https://arxiv.org/html/2508.08221)
- SimpleRL-Zoo: KL loss coefficient 1e-4 for 0.5B to 14B. — [SimpleRL-Zoo, arXiv 2503.18892](https://arxiv.org/html/2503.18892)
- CMU 100M from-scratch models: low-variance KL penalty, coefficient 1e-3. — [Interplay of Pre-/Mid-Training and RL, arXiv 2512.07783](https://arxiv.org/html/2512.07783)
- Skywork-OR1 (7B/32B): KL beta = 1e-3 pulled the policy back to the reference, KL went toward zero, and AIME24 gains stalled. They set beta = 0 for all released stages. — [Skywork-OR1, arXiv 2505.22312](https://arxiv.org/html/2505.22312)
- ProRL (1.5B, starting from DeepSeek-R1-Distill-Qwen-1.5B) keeps a KL penalty because the starting checkpoint already produces coherent CoT: "retaining a KL penalty is still beneficial for both stability and sustained entropy". It hard-resets the reference policy and optimizer state when validation stagnates or degrades, with no fixed interval (resets at Run 2 and twice across Runs 6–7). — [ProRL, arXiv 2505.24864](https://arxiv.org/html/2505.24864)
- In Cui et al., a reference KL stabilized entropy but degraded performance (Qwen2.5 family). — [Entropy Mechanism, arXiv 2505.22617](https://arxiv.org/html/2505.22617)

### Inferences
- Our run starts from SFT, so it matches the ProRL/CMU regime more than the R1-Zero regime. A small KL (1e-3, low-variance "k3" estimator) is defensible as a guard against drift and degeneration at tiny scale, where capacity is scarce. Skywork and Cui suggest it caps gains. A pre-registration could fix KL=1e-3 for the main arm, or run KL=0 vs 1e-3 as two seeds of a factor, with ProRL-style resets triggered on held-out-val plateau.

### Gaps
- The ProRL KL coefficient value was not in the extracted text.

## Entropy collapse: measurements, remedies and training temperature

### Takeaway
Entropy collapses early: in Cui et al., about 73% of entropy loss and 76% of the performance gain occur in the first 200 of 2,400 steps. Performance then follows R = -a*exp(H) + b, so it saturates once entropy is gone. Remedies with measured gains are Clip-Cov and KL-Cov (+2.0 at 7B, +6.4 at 32B), adaptive entropy control (target 0.2), clip-higher (0.28 to 0.4), and training temperature around 1.0 or higher. Temperature 0.6 caused collapse within about 100 steps. Fixed entropy bonuses are fragile.

### Cited Findings
- Empirical law R = -a*exp(H) + b, with a and b fitted, unchanged across GRPO, RLOO and PRIME, and log-linear in Qwen2.5 model size. Fitting the first 36 steps predicts the next 200 steps within about 1% RMSE. — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- About 73% of entropy loss and 76% of performance gain occur in the first 200 gradient steps. The first 800 steps give over 93% of the gains (11 base models, 0.5B–32B). — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- Mechanism: entropy change is approximately the negative covariance between a token's log-probability and its logit change, which under policy gradient tracks probability x advantage. High-probability, high-advantage tokens drive collapse. — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- Clip-Cov detaches gradients for a random fraction of tokens with covariance in [1, 5], at clip ratio 2e-4. KL-Cov applies KL with beta=1 to the top-k fraction of tokens by covariance, with k = 2e-3 (7B) or 2e-4 (32B). KL-Cov vs GRPO: 40.6 vs 38.6 at Qwen2.5-7B, and 52.2 vs 45.8 at 32B. — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- An entropy-loss coefficient of 0.01 made entropy explode, and 0.005 stabilized it without beating baselines. Clip-higher at 0.28 raised entropy early but became unstable later. — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- Cui et al. training setup: LR 5e-7, 256 prompts x 8 responses at temperature 1, 8 policy updates per rollout, max length 8,192. — [Entropy Mechanism](https://arxiv.org/html/2505.22617)
- Skywork-OR1: faster entropy collapse goes with worse test performance. At temperature 0.6, entropy started very low or collapsed within about 100 steps and learning stagnated. At 1.0, learning was stronger. Adaptive entropy control applies entropy loss only when entropy falls below a target (tgt-ent = 0.2), adjusting the coefficient by Δ = 0.005 per step. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
- ProRL (1.5B): rollout temperature 1.2 and eps_high = 0.4. Raising temperature only delayed collapse. DAPO components plus KL avoided collapse. — [ProRL](https://arxiv.org/html/2505.24864)
- POLARIS: pick the training temperature where accuracy starts to decline while diversity (distinct 4-gram ratio) is still high. Initial values were 1.4 for Qwen3-4B and 0.7 for R1-Distill-7B. Temperature rose by stage (4B: 1.4, 1.45, 1.5; 7B: 0.7, 1.0, 1.1) to track the entropy drop. No entropy loss was used, to avoid instability. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris/)
- Tricks or Traps: clip-higher had little effect on base models (clip rate about 0.003) but slowed collapse and helped aligned models. The best eps_high was 0.32 for aligned Qwen3-4B and 0.28 for aligned 8B. — [Tricks or Traps](https://arxiv.org/html/2508.08221)

### Inferences
- An SFT'd tiny model with pass rates near 0 or 1 is effectively already "collapsed" per prompt. Remedies that raise exploration, such as higher temperature (1.0–1.2 chosen by a POLARIS-style sweep on the pass-rate distribution) and eps_high 0.28–0.4, act directly on our 20% mixed-group problem. For a tiny, aligned model, Tricks or Traps' finding that clip-higher helps aligned models more (with a larger optimum at 4B than at 8B) weakly supports eps_high around 0.3 or higher.
- Because about 75% of the gain arrives in the first ~200 steps, the confirmatory run should track and log policy entropy per step. A pre-registered entropy floor (Skywork-style target) is cheaper than tuning a fixed bonus.

### Gaps
- None of these entropy measurements were made below 0.5B. I did not verify the Clip-Cov and KL-Cov fractions for small models.

## Learning rate, batch size, epochs/minibatches, steps and group size, and scaling down

### Takeaway
RL LRs at 0.5B–32B are nearly always 5e-7 to 2e-6, constant, with short warmup. The only 100M from-scratch study also used 1e-6, with batch 1,024, KL 1e-3, temperature 1.0, about 6 rollouts per prompt and roughly 2 epochs over 200K samples. Group sizes of 8 to 16 dominate, with 64 in DeepSeekMath. ScaleRL found 8 vs 32 generations per prompt essentially equivalent at a fixed total batch. Our previous LR of 1e-5 is 5–10x above every recipe found.

### Cited Findings
| Source (model) | LR | Prompts/batch x G | Updates per rollout | Temp | KL | Steps |
|---|---|---|---|---|---|---|
| [DeepSeekMath](https://arxiv.org/html/2402.03300) (7B) | 1e-6 | batch 1024, G=64 | 1 | – | 0.04 | – |
| [DAPO](https://arxiv.org/html/2503.14476) (32B) | 1e-6, 20-step warmup | 512 x 16 | 16 | – | 0 | – |
| [Dr. GRPO](https://arxiv.org/html/2503.20783) (1.5–7B) | 1e-6 | G=8 | 1 epoch | 1.0 | 0 | ~1 day 8xA100 |
| [ProRL](https://arxiv.org/html/2505.24864) (1.5B) | 2e-6 const | 256 x 16 (32 in later runs) | 4 | 1.2 | kept | >2k |
| [Entropy Mechanism](https://arxiv.org/html/2505.22617) (0.5–32B) | 5e-7 | 512x8 (Sec 2), 256x8 (Sec 4) | 8 (Sec 4) | 1.0 | 0 | 2,400 |
| [Tricks or Traps](https://arxiv.org/html/2508.08221) (4B/8B) | 1e-6, 50-step warmup | 128 x 8 | 1 epoch | 0.99 | 0 | 500 |
| [SimpleRL-Zoo](https://arxiv.org/html/2503.18892) (0.5–14B) | n/r | 1024 x 8, mini-batch 256 | 4 | 1.0 | 1e-4 | n/r |
| [ScaleRL](https://arxiv.org/html/2510.13786) (8B) | 5e-7, 100-step warmup | up to 2048 prompts | async, k=8 | – | – | – |
| [CMU 100M](https://arxiv.org/html/2512.07783) (100M from scratch) | 1e-6 | global 1024, mini 256, r=6 | 4 (implied by 1024/256) | 1.0 | 1e-3 | 2 epochs of 200K, or 100 steps |
| [TRL defaults](https://huggingface.co/docs/trl/main/en/grpo_trainer) | 1e-6 | G=8 | num_iterations=1 | 1.0 | 0 | – |

- SimpleRL-Zoo ran Qwen2.5-0.5B on medium-difficulty data and raised the average from 12.1 to 20.9 (MATH500 from 15.8 to 34.4). Data difficulty had to match capability: harder data caused collapse (length growth without accuracy) on weaker models. G=1 or 4 caused instability in an ablation (Mistral-24B). — [SimpleRL-Zoo](https://arxiv.org/html/2503.18892)
- Skywork-OR1: in on-policy runs, larger rollout batch and group size had only minor effects on entropy dynamics, while section headings claim that larger batch and group sizes improve test performance. I could not read the evidence in the body text. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
- ScaleRL: sweeping 8 to 32 generations per prompt at a fixed total batch left the fitted curves essentially unchanged at moderate batch sizes. Larger batches are slower early but reach a higher asymptote. — [ScaleRL](https://arxiv.org/html/2510.13786)
- POLARIS: drop prompts solved 8/8 offline, which gives a "mirrored J" difficulty distribution. Remove prompts above 0.9 accuracy after each stage. "Rollout Rescue" swaps a stored correct response into 0/8 groups, and zero-advantage samples are replaced by duplicates of mixed groups. G=8. — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris/)
- CMU 100M: RL produces real capability gains only when RL data sit at the "edge of competence" (fails pass@1, solves pass@k). RL on in-distribution tasks improved pass@1 but not pass@128. Calibrated edge data gave up to +42% pass@128 out of distribution. Mid-training plus RL beat RL alone by +10.8% on OOD-hard. With full mid-training, light RL (β=0.2 of compute) gave the best pass@1 on OOD-edge tasks. — [CMU, arXiv 2512.07783](https://arxiv.org/html/2512.07783)
- DeepSeekMath (7B): RL improved Maj@K but not Pass@K. RL gains over Instruct were GSM8K 82.9 to 88.2 and MATH 46.8 to 51.7. — [DeepSeekMath](https://arxiv.org/html/2402.03300)

### Inferences
- Scaling down: the CMU 100M study is closest to our setting (from scratch, Qwen-style, synthetic arithmetic, pretrain LR 2e-4), and it used RL LR 1e-6, the same as 1.5B–32B recipes. Nothing found supports raising the RL LR at small scale. Our 1e-5 is a plausible cause of losing to SFT. A pre-registered LR of 1e-6 to 2e-6 (optionally a 2-point sweep {1e-6, 3e-6} on val) is the evidence-backed choice.
- Group size: with per-prompt pass rate p, P(mixed group) = 1 - p^G - (1-p)^G. For p = 0.05 this is 0.34 at G=8, 0.56 at G=16 and 0.81 at G=32. For p = 0.95 it is symmetric. Raising G from 8 to 16–32 therefore mostly rescues prompts with rare successes or rare failures. Rollouts are only 76–256 tokens and the model is 43M parameters, so G=16–32 is cheap. ScaleRL suggests the total batch matters more than G, which supports keeping prompts x G at 1,024 or more.
- Data selection is likely the larger lever. Pre-filter the RL pool to prompts with offline pass rate strictly between 0 and 1 (POLARIS, CMU "edge of competence", ScaleRL No-Positive-Resampling at 0.9 or higher). If only about 20% of prompts are mixed, an unfiltered pool spends about 80% of rollout compute on zero-gradient groups.
- DeepSeekMath and CMU both report that RL mainly sharpens pass@1 toward what pass@k already covers. If the SFT model's per-prompt pass rates are already near 0 or 1, pass@1 headroom is small unless training targets the edge band. This may explain why time-matched extra SFT won.
- Step count: ~75% of gains in the first 200 steps (Cui et al.), and the CMU study used about 100–400 steps at batch 1,024. A pre-registration of a few hundred steps with fixed checkpoint selection on held-out val is consistent with this.

### Gaps
- I found no published LR or group-size sweep below 100M.
- The CMU paper's clip eps and PPO epoch count were not in the extracted text.

## Off-policy staleness, importance ratios and gradient steps per batch

### Takeaway
Skywork-OR1 directly measured that more SGD steps per rollout batch speeds up entropy collapse and hurts test performance, and identified off-policy data itself as the cause. Recipes vary from 1 update per batch (DeepSeekMath, Dr. GRPO, Tricks or Traps, TRL) to 16 (DAPO). For a tiny model, on-policy updates (1 per batch) is the conservative choice.

### Cited Findings
- N_SGD = (rollout batch / mini-batch) x N_reuse. Raising N_SGD, by shrinking mini-batches or reusing data, drove entropy to near zero within a few steps and stopped test improvement. On-policy runs with the same mini-batch size did better over time. A larger rollout batch did not prevent collapse off-policy, and may have sped it up (7B/32B). — [Skywork-OR1](https://arxiv.org/html/2505.22312)
- Clip-higher slowed and stabilized entropy, but entropy still fell faster than in fully on-policy training. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
- ScaleRL: PipelineRL and PPO-off-policy reached similar asymptotes. A maximum off-policyness of k=8 was optimal for throughput. FP32 LM-head logits raised the asymptotic reward from about 0.52 to 0.61, by reducing train/inference probability mismatch. — [ScaleRL](https://arxiv.org/html/2510.13786)
- With num_iterations=1, the clipped objective reduces to the plain policy gradient (TRL docs, following DeepSeekMath). — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- GSPO notes that sequence-level likelihoods could use inference-engine log-probs directly, avoiding recomputation. — [GSPO](https://arxiv.org/html/2507.18071)

### Inferences
- If rollouts come from a separate inference path (KV-cache sampler in bf16) and the loss is computed in the training graph, a ScaleRL-style FP32 logit head and recomputed log-probs avoid a silent importance-ratio mismatch. This is cheap to check: on-policy ratios should be 1.0 ± small.
- With 1 update per batch, eps_low and eps_high only bind if there is mismatch. Clip-higher matters only when N_SGD > 1. A pre-registration should state N_SGD explicitly. If N_SGD = 2–4 (as in CMU and ProRL) is used for sample efficiency, pair it with eps_high 0.28–0.4.

### Gaps
- I found no study of staleness effects at sub-1B scale.

## Evidence on RL for models under 100M parameters

### Takeaway
I found one rigorous study near our scale: CMU's 100M from-scratch models on GSM-Infinite-style synthetic arithmetic (ICML 2026 spotlight). It shows that GRPO helps only on edge-of-competence data, and that mid-training plus light RL beats RL alone. Otherwise there are only tutorials and toy repos (for example a 13M LLM with GRPO), with no rigorous results.

### Cited Findings
- CMU 100M architecture: 12 layers, d=768, FFN 3,072, 12 heads / 2 KV heads, vocab 2,200, pretraining on 10B tokens. RL: GRPO, LR 1e-6, global batch 1,024, mini-batch 256, KL 1e-3, entropy bonus 0, temperature 1.0, r=6 rollouts per prompt, max length 1,024. — [CMU, arXiv 2512.07783](https://arxiv.org/html/2512.07783)
- Findings: RL on in-distribution difficulty improves pass@1 but not pass@128. Edge-of-competence data (op count 11–14 where base pass@128 is still nonzero) gives up to +42% pass@128 out of distribution. Mixing process-verification reward (0.2 outcome / 0.8 process, or a strict variant granting outcome reward only when the process verifies) improves pass@1 by 4–5% on op 15–20. — [CMU](https://arxiv.org/html/2512.07783); [ICML 2026 listing](https://icml.cc/virtual/2026/spotlight/84767)
- Toy implementations of GRPO on tiny from-scratch models exist, for example a Countdown-style arithmetic GRPO in pure PyTorch with SFT warm-start, and a from-scratch LLM repo whose smallest model is 13M with GRPO and an arithmetic curriculum. These report no rigorous comparisons against SFT. — [rl-fine-tuning-from-scratch](https://github.com/Anirudh-Senani/rl-fine-tuning-from-scratch-build-the-environment-train-with-grpo); [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch); [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)
- The smallest model in mainstream RLVR papers is Qwen2.5-0.5B (SimpleRL-Zoo, Cui et al.). It does gain from RL (average 12.1 to 20.9), but only on difficulty-matched data. — [SimpleRL-Zoo](https://arxiv.org/html/2503.18892)

### Inferences
- The CMU result is the strongest guide: RL beats SFT-only approaches mainly when the RL pool is restricted to the model's edge of competence. Our arithmetic setting allows a process reward, since intermediate steps in show-your-work traces are verifiable. That is a documented +4–5% pass@1 lever at 100M, but changing the reward away from 0/1 exact-answer would be a protocol change.
- Suggested pre-registerable configuration, synthesized from the above:
  - Advantage and loss: Dr. GRPO-style. Use group mean (or RLOO) with no std or batch-level std, and a constant-denominator loss.
  - Updates: on-policy, 1 update per batch, or at most 2–4 with eps_low 0.2 and eps_high 0.28–0.3.
  - Optimizer: LR 1e-6 constant (or 2e-6), short warmup of 10–20 steps, gradient clip 1.0.
  - Sampling: G = 16 (or 32), at least 1,024 samples per step, temperature 1.0 (chosen by a POLARIS-style diversity sweep, possibly 1.1–1.2).
  - KL: 1e-3 low-variance estimator, or 0. ProRL-style reset on val plateau.
  - Data: pool pre-filtered to offline pass rate in (0, 1), with prompts at 0.9 or higher dropped as training proceeds. Keep dynamic sampling.
  - Steps and monitoring: a few hundred steps with fixed val-based checkpoint selection, FP32 logits for log-prob computation, and per-step entropy logging.
  - Caveat: this is an inference from 100M–32B evidence, not a validated recipe at 43M.

### Gaps
- I found no peer-reviewed RL results below 100M parameters.
- I found no study comparing RL against a compute-matched extra-SFT control at small scale. The CMU study compares mid-training against RL compute splits, not against continued SFT on the SFT data.
