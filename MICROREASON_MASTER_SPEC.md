# MicroReason: master specification and agent handoff

Version 1.0 | 7 November 2026 | Proposed project design

This is the execution brief for an agent starting with an empty project folder. No model, dataset, benchmark, training run, checkpoint, or API is implemented or verified yet. This document consolidates the accessible MicroReason design from the referenced ChatGPT conversation. The conversation reader truncated the long answer in its budget section, so later details are explicit implementation proposals. **Targets and estimates are not measured results.**

## 1. Purpose and research question

MicroReason is a small decoder language model trained from random initialization on licensed text and generated mathematics. It covers tokenization, base pretraining, mathematical continued pretraining, supervised fine-tuning (SFT), Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), evaluation, and local inference.

The research question is whether a small model can acquire measurable arithmetic and basic algebra capability through these stages within a bounded compute budget. The deliverable is a reproducible experiment with stage checkpoints and honest negative results. Do not present the model as a general reasoning system or claim that generated steps faithfully describe its internal computation.

| ID | Proposed hypothesis | Primary comparison |
| --- | --- | --- |
| H1 | Math mid-training improves math capability. | v1-math against v0-base. |
| H2 | SFT improves instruction following and answer format. | v2-sft against v1-math. |
| H3 | DPO improves response selection or formatting without a major correctness loss. | v3-dpo against v2-sft. |
| H4 | Exact-outcome GRPO improves held-out correctness beyond SFT and equal-cost continued SFT. | v4-grpo against v2-sft and v2-extra-sft. |

These are hypotheses, not expected findings. Explicit non-goals are general chatbot quality, difficult GSM8K or MATH scores, multimodal input, tool use, a billion-parameter model, human preference labeling, a learned reward model, and production inference.

## 2. Decisions, provenance, and invariants

**Inherited design.** The referenced conversation proposed a 12-layer, 512-wide causal Transformer with 8 attention heads, RoPE, RMSNorm, SwiGLU, an 8,192-ID byte-level BPE vocabulary, tied embeddings, and a 2,048-token context. It proposed 120 million base tokens, 40 million mid-training tokens, 50,000 SFT examples, up to 20,000 DPO pairs, a 20,000-prompt RL pool, an RTX 4090 for first GPU runs, and roughly $50 as an initial compute cap. None of these runs was reported as completed.

**Resolved choices.** Use one tokenizer across all stages. DPO and GRPO are independent branches from v2-sft. Use v2-extra-sft as an equal-compute control. The primary GRPO reward is binary exact-answer correctness. Freeze generator and evaluation manifests. Names and commands in this file are contracts for the implementing agent, not working commands today.

**Unknown until measured.** Throughput, memory, actual cost, useful curriculum difficulty, learning-rate stability, GRPO reward variance, and effect sizes require smoke tests or experiments.

The implementation must maintain these invariants.

1. Freeze the test manifest before any training. Do not feed its examples into training, preference construction, RL, prompt tuning, or error-driven fixes.
2. Give every generated record a stable ID, generator version, split, seed, problem family, exact answer, and verifier specification.
3. Bind every checkpoint to a config hash, tokenizer hash, data manifest hashes, code revision, and parent checkpoint.
4. Resume without silently repeating or skipping records. Save model, optimizer, scheduler, random states, sampling state, step, and consumed-token count.
5. Do not start paid GPU use, paid data access, external publication, or public deployment without separate explicit bounded authorization. A budget in this file does not authorize spending.

## 3. Model architecture

| Component | Proposed v1 setting |
| --- | --- |
| Objective | Causal next-token prediction. Later stages adapt the same weights. |
| Blocks | 12 decoder blocks, pre-norm. |
| Hidden width | 512. |
| Attention | 8 query heads and 8 KV heads, each 64 wide. |
| Position | Rotary position embeddings on query and key. |
| MLP | SwiGLU with 1,408 intermediate channels. |
| Norm | RMSNorm before attention and MLP, plus final RMSNorm. |
| Vocabulary | 8,192 IDs including special tokens. |
| Context | Maximum 2,048 tokens; begin base training at 1,024. |
| Embeddings | Input and output weights tied. |
| Projections | No biases in attention and MLP. |
| Dropout | 0.0 for baseline. |
| Compute | BF16 on a supported GPU; FP32 optimizer states. Use compatible types for CPU or MPS smoke tests. |

Use PyTorch optimized scaled-dot-product attention where supported. Verify masking, padding, and RoPE positions in tests. Enable gradient checkpointing only if a measured memory need warrants it. Do not add custom CUDA kernels in v1.

**Parameter arithmetic.** Tied token embeddings contribute 8192 × 512 = 4,194,304 weights. Each block contributes four 512 × 512 attention matrices, three 512 × 1408 SwiGLU matrices, and two 512-channel norms. Twelve blocks plus final norm yield about **42,742,272 parameters**. The prior conversation's "about 43.5M" was a rough estimate. Print the actual instantiated count and reconcile any difference before full training. Added biases, learned position scales, or untied output weights require a revised count.

Parameterize the same model class at 2 to 5 million parameters for smoke tests. A different smoke-test architecture could hide bugs in the full model.

### Tokenizer and response contract

Train an 8,192-ID byte-level BPE tokenizer on the licensed text and synthetic math training mixture. Reserve unique, single-token IDs for [pad], [bos], [eos], [user], [assistant], [steps], [/steps], [answer], and [/answer]. Record normalization, pre-tokenization, special IDs, vocabulary, merges, training manifest, and hash. Keep token IDs fixed across all checkpoints.

Use UTF-8 text, canonical line endings, and a documented Unicode normalization rule. Test round trips for signs, integers, decimals, fractions, parentheses, and equations. Include -12, 12, 1/2, 0.5, and x = -3/4 in a tokenizer diagnostic. A digit-level number tokenizer is an optional separate from-scratch ablation because changing token IDs invalidates checkpoint comparison.

The versioned SFT format is:

~~~text
[bos][user]Solve 3(x + 4) = 27. Show your work.[eos][assistant]
[steps]
1. Divide both sides by 3: x + 4 = 9.
2. Subtract 4: x = 5.
[/steps]
[answer]5[/answer][eos]
~~~

Centralize prompt assembly. SFT loss applies only to assistant tokens, including the answer and end marker. Ignore user and padding labels. For direct-answer evaluation, use a separately versioned prompt requesting only an answer block.

## 4. Data generation and isolation

### Record schema

Store generated problems as versioned JSONL or Parquet. Each record contains id, family, template_id, difficulty, prompt, solution_steps, answer_type, final_answer, verifier_spec, generator_version, seed, split, and a semantic fingerprint. Store source-corpus records in a separate schema with source, license, retrieval date, document ID, split, and checksum. Source metadata is not model input.

~~~json
{
  "id": "linear_one_side_v1_000142",
  "family": "linear_equation",
  "template_id": "ax_plus_b_equals_c",
  "difficulty": 2,
  "prompt": "Solve for x: 4x + 7 = 31",
  "solution_steps": ["Subtract 7: 4x = 24", "Divide by 4: x = 6"],
  "answer_type": "rational",
  "final_answer": "6",
  "verifier_spec": {"kind": "linear_equation", "coefficients": [4, 7, 31]},
  "generator_version": "1.0",
  "seed": 142,
  "split": "train",
  "semantic_fingerprint": "sha256-of-canonical-problem"
}
~~~

Generate exact answers with Python integers and fractions.Fraction. Use SymPy as an independent checker for symbolic families, with timeouts and a restricted grammar. Never run eval, exec, or arbitrary expressions from model output. Reject records when two checking methods disagree, division is undefined, the prompt is ambiguous, or the answer has multiple meanings under the task grammar.

### Curriculum and quantities

These are **generation and processed-token targets**, not guarantees that the compute cap can process them. Expose max_train_tokens, max_steps, and max_examples separately so pilots can use the same dataset definitions.

| Dataset | Initial target | Purpose |
| --- | ---: | --- |
| Base corpus | 120 million processed tokens | Generic text plus math expressions. |
| Math mid-training | 40 million processed tokens | Expressions, problems, and solution traces. |
| SFT | 50,000 examples, 2 proposed epochs | Instruction response training. |
| DPO | Up to 20,000 pairs, 1 proposed epoch | Preference comparison. |
| RL prompt pool | 20,000 unique prompts | On-policy rollouts. |
| RL completions | 20,000 to 40,000 initially | Planning range subject to budget and reward variance. |
| Key validation and test suites | 5,000 to 10,000 problems each where practical | Paired comparison and failure analysis. |

Use five difficulty levels. Level 1 covers signed addition, subtraction, and multiplication. Level 2 adds exact rational division, order of operations, and one-step equations. Level 3 adds multi-step arithmetic and ax+b=c. Level 4 adds parentheses and equations with variables on both sides. Level 5 holds out selected compositions, larger values, and longer traces. A family may span levels; record both fields. Avoid unsolvable or non-unique equations unless the task explicitly asks for a set-valued answer and the verifier supports it.

For base pretraining, **propose** 70% licensed compact general text and 30% generated mathematics by processed tokens. The source conversation specified a general-plus-math mixture but no ratio. Freeze this ratio in the config before the main run. Mid-training uses the inherited 85% math and 15% general-text replay mixture. Compare curriculum to uniform sampling only after the baseline stages fit the budget.

Use only a general-text source with a recorded license or explicit permission for the intended training and redistribution. Record URLs and checksums in a manifest. Do not copy benchmark test problems into the corpus. Generate synthetic data from a root seed and stable IDs so a change in shuffle order does not alter content.

### Splits and leakage checks

Reserve whole **template families** for the held-out-template test before generating rows. For train-visible families, split records by disjoint seeds into train, validation, and in-distribution test. Use a distinct number range for extrapolation tests. Keep difficulty-stratified tests separate. Hash the frozen test manifest and keep it outside training glob patterns.

Deduplicate exact text and canonical semantic forms across splits. Canonicalize equivalent coefficient tuples, commutative operand order where appropriate, whitespace, and trivial rearrangements. Scan the general corpus for n-gram overlap with generated test prompts. Treat collisions as contamination and fix the generator or split. Never promote a test failure into SFT, DPO, or RL data. Dev errors may guide a declared versioned intervention; seal the final test until the analysis plan is fixed.

Data build gates:

- The same generator version and seed produce byte-identical records and hashes.
- Every answer passes the independent verifier. A sample of traces receives a separate step check.
- Mutated answers, equivalent fractions, malformed markers, and adversarial strings exercise the verifier.
- A report prints family, difficulty, length, number-range, source, and token counts by split.
- No IDs, fingerprints, or normalized prompts overlap across train and held-out validation or test partitions. DPO pairs may reuse training prompts, but DPO train and validation prompts must be disjoint and no preference prompt may occur in the final test.

## 5. Exact answer verification and reward

The generator owns the expected answer. The verifier is a separate pure module that accepts a prompt record and an untrusted completion. It returns a typed result with parse status, canonical answer, exact correctness, and a reason code. It must not consult model logits, train-set membership, or an LLM judge.

For v1, the answer grammar permits an integer or a rational written as p/q, with an optional leading minus sign. Strip surrounding whitespace inside one [answer] block. Reduce fractions with exact arithmetic and normalize denominator sign. Accept mathematically equivalent forms such as 2/4 and 1/2. Do not silently round decimals. A later decimal task must define precision and tolerance in its record schema before data generation. For equation tasks, verify the canonical rational against the generated solution and substitute it into the original equation with exact arithmetic. For algebraic expressions, parse only a restricted symbolic grammar and compare normalized expressions under stated assumptions.

The response parser requires exactly one well-formed [answer]...[/answer] block. It rejects missing, duplicate, nested, excessively long, nonnumeric, or out-of-grammar answers. It ignores text outside the block for the primary reward, but logs whether the optional [steps] block is well formed. Reject model output that tries to impersonate the verifier, insert additional role markers, or exploit parser delimiters. Put a length and execution-time bound on every parse and symbolic check. Verifier errors fail closed and appear as a separate metric from incorrect answers.

The primary GRPO reward is:

~~~text
R_primary(problem, completion) = 1 if the response parses and its exact answer is correct; otherwise 0.
~~~

Do not reward response length or plausible-looking reasoning in the first experiment. An optional, explicitly separate ablation can use R_shaped = R_correct + 0.1 × R_format, where R_format is 1 only for a valid single answer block. It must report both correctness and format separately because the shaped reward can improve format without improving mathematics. Do not use the shaped reward to claim success for H4's primary result.

The verifier test suite must include correct and wrong integers, unreduced and negative fractions, negative denominators, zero denominators, duplicate answer blocks, markup injection, Unicode minus, overflow-sized numbers, symbolic equivalences, and timeouts. Test that a completion with a correct number in [steps] but a wrong [answer] receives zero.

## 6. Training stage graph and configs

The stage graph is a directed experiment graph. Save an immutable checkpoint at each node:

~~~text
random weights
  → v0-base
  → v1-math
  → v2-sft
       ├→ v2-extra-sft      additional-compute control
       ├→ v3-dpo            preference branch
       └→ v4-grpo           primary RL branch
             optional: v3-dpo → v4-dpo-grpo
~~~

The optional DPO-to-GRPO branch answers a separate question. Do not make it the primary GRPO comparison. Keep optimizer state separate across branches. All branches share the same v2-sft parent weights and frozen evaluations. A branch name denotes training history, not a quality rank.

### Stage 0: infrastructure and no-cost proof

Implement tokenizer, model, generator, verifier, loader, training loop, evaluator, checkpoints, and CLI before a paid run. The loop needs gradient accumulation, clipping, logging, a scheduler, periodic evaluation, atomic checkpoint writes, and deterministic resume. Run CPU or MPS smoke tests with a 2 to 5 million parameter model. Overfit one tiny batch to near-zero loss. Resume a short run and compare its weights, optimizer state, consumed record IDs, and next-step loss against an uninterrupted run within a declared numerical tolerance. Verify attention causality by changing future tokens and checking earlier logits. Print parameter counts and memory estimates. **Full pretraining cannot start until these tests pass.**

### Stage 1: base pretraining

Start v0-base from random weights. Use causal cross-entropy on packed sequences, with explicit loss masking for padding and document boundaries as the chosen packing policy requires.

| Setting | Initial proposal |
| --- | --- |
| Optimizer | AdamW. |
| Peak learning rate | 3 × 10^-4. |
| Weight decay | 0.1, excluding norm and embedding parameters if the implementation's declared policy says so. |
| Warmup | 2% of planned optimizer steps. |
| Schedule | Cosine decay to 10% of peak. |
| Sequence length | 1,024 initially. |
| Effective batch | 65,536 training tokens. |
| Gradient clip | Global norm 1.0. |
| Processed-token target | 120 million. |
| Evaluation | Every 100 to 250 optimizer steps. |
| Checkpoint | Every 250 to 500 optimizer steps and at stage end. |

120,000,000 ÷ 65,536 = 1,831.055, so reaching at least the target requires **1,832 full-token-equivalent steps**. The earlier "approximately 1,831" was rounded. Exactly 1,831 full batches contain 119,996,416 tokens; log the actual count, including any partial batch. Track general-text and math validation loss separately. A gate to proceed is finite gradients, declining validation loss relative to the first evaluation, stable checkpoint resume, and completion behavior better than random initialization. A loss decline is an engineering gate, not evidence of reasoning.

### Stage 2: mathematical mid-training

Continue v0-base with the same causal LM objective. Use 85% generated math and solution traces and 15% licensed general-text replay by processed tokens. Move from easy arithmetic to multi-step expressions and algebra with a logged sampler schedule. Keep a uniform-sampling ablation separate.

| Setting | Initial proposal |
| --- | --- |
| Parent | v0-base. |
| Peak learning rate | 1 × 10^-4. |
| Warmup | 2%. |
| Effective batch | 32,768 tokens. |
| Context | 1,024 initially; 2,048 only after a memory and throughput check. |
| Processed-token target | 40 million. |
| Optimizer and clipping | AdamW and global norm 1.0 unless the pilot requires a documented change. |

40,000,000 ÷ 32,768 = 1,220.703, so the full-token-equivalent target takes **1,221 steps**. Track math loss, generic-text loss, and task accuracy against v0-base. General replay is intended to reduce forgetting; measure that claim. Save v1-math even if the result is negative. If math accuracy does not rise, inspect data quality and task fit before changing the curriculum.

### Stage 3: supervised fine-tuning

Continue v1-math on 50,000 generated, independently verified instruction-response examples for a proposed 2 epochs. Use a starting learning rate of 5 × 10^-5 and maximum sequence length of 512 to 1,024. Mask the user prompt from loss. Train both step-by-step examples and direct-answer examples only if their mixing weights are declared and fixed before the run. Do not truncate the final answer block. Evaluate direct-answer and show-work prompts separately for accuracy, parse rate, and token count.

Save v2-sft. The engineering exit gate is that the model can generate the documented response format at a high rate and solves some easy held-out problems. The exact rate is a declared target, not a requirement to alter the test. If v2-sft has near-zero success on the proposed RL pool, reduce the RL task difficulty or improve SFT using train and validation data before GRPO.

### Stage 4A: DPO preference branch

Generate up to 20,000 chosen/rejected pairs from the same prompt. Chosen responses must be verified correct and well formed; rejected responses must have a verified wrong answer, invalid format, or demonstrable step error under a separate step checker. Prefer a substantial proportion of responses sampled from v2-sft over trivial synthetic corruptions. Record the generation checkpoint, sampling parameters, verifier result, pair selection reason, and prompt ID. Exclude ambiguous pairs and pairs whose chosen/rejected distinction depends on subjective prose quality.

Use v2-sft as both the policy initialization and a frozen reference. Proposed DPO beta is 0.1, LR pilot range is 1 × 10^-6 to 5 × 10^-6, and training is 1 epoch. Compute sequence log probabilities only over assistant response tokens. Keep both responses to a prompt in the same data split. Assess held-out pair discrimination, answer accuracy, valid format, and length. Save v3-dpo even if it regresses; do not use it as the default RL parent.

For a pair (chosen, rejected), use the standard DPO log-sigmoid objective on the difference between policy and frozen-reference log-probability margins. Verify sign conventions on a tiny synthetic example before a full run. A high pair-classification score alone does not establish mathematical improvement.

### Stage 4B: equal-compute SFT control

Continue v2-sft on additional **training-only** supervised examples. Match v4-grpo by measured incremental GPU time, including generation, verification, training, and evaluation overhead as defined before either run. Report both GPU time and processed tokens because the training objectives spend compute differently. Choose a fixed control recipe before reading final test results. Save v2-extra-sft. If a precise time match is impossible, report the mismatch rather than claiming equality.

### Stage 5: on-policy GRPO branch

Start v4-grpo directly from v2-sft. Draw prompts from the RL training pool. For each prompt, sample a group of 4 completions from the current policy at temperature 0.8, with at most 128 new tokens per completion. Verify each completion with the exact-answer verifier. Standardize rewards within each group to form relative advantages. A group with zero reward variance has zero informative relative advantage; log it and do not invent a signal. Keep an immutable rollout record containing prompt ID, policy checkpoint, sampled token IDs, log probabilities, reward result, and generation settings.

Initial settings are LR 1 × 10^-6, PPO-style ratio clip epsilon 0.2, group size 4, 20,000 to 40,000 completions as a planning range, and evaluation every 100 to 200 optimizer steps. Start with zero explicit reference-KL coefficient only if the implementation's objective and implicit regularization are documented. Run nonzero KL as an ablation if the first run shows policy drift. The actual optimizer-step count depends on batch size, group scheduling, and generated length; do not infer it from the completion count.

The objective must use log probabilities from the policy that generated each rollout and the current policy's response-token log probabilities. State how old-policy ratios, clipping, group normalization, KL, truncation, and length normalization are implemented. Never mix rollout groups from incompatible policy checkpoints without declaring a bounded staleness rule. Mask prompt and padding tokens from the policy loss. Use a small analytical test where two responses have known rewards to verify advantage signs and gradient direction.

Before the full run, sample at least 200 prompts from the **RL training distribution** and estimate pass@1, the fraction of groups with both successes and failures, and the fraction of malformed answers. The numerical pilot size is a proposed engineering choice. If almost all groups are all-wrong or all-correct, adjust the training-only prompt difficulty or SFT model. Do not "fix" zero success by spending more on RL. Log mean reward, success rate by family, mixed-group fraction, KL if used, response length, clip fraction, gradient norm, verifier errors, and cost. Save v4-grpo and evaluate all branches under identical final protocols.

An optional v3-dpo → v4-dpo-grpo branch may test DPO initialization after the primary branch and budget review. It is not part of the minimum experiment.

### Config discipline

Use versioned YAML or TOML for each stage and a resolved JSON snapshot per run. An illustrative stage config is:

~~~yaml
stage: pretrain
model: configs/model_43m.yaml
tokenizer: artifacts/tokenizer/v1
data_manifest: data/manifests/base_train_v1.json
validation_manifests:
  - data/manifests/general_val_v1.json
  - data/manifests/math_val_v1.json
seed: 42
precision: bf16
sequence_length: 1024
effective_batch_tokens: 65536
max_train_tokens: 120000000
optimizer:
  name: adamw
  peak_lr: 0.0003
  weight_decay: 0.1
  grad_clip_norm: 1.0
scheduler:
  name: cosine
  warmup_fraction: 0.02
  min_lr_fraction: 0.1
budget:
  paid_compute_allowed: false
  max_spend_usd: 0
~~~

Validate config schema before training. The default must disallow paid compute. A run manifest records all resolved defaults, software versions, GPU model, system memory, hashes, seed, start and end times, processed tokens, examples, steps, GPU-hours, and cost estimate. Never overwrite a checkpoint in place; write to a temporary path, verify checksum, then atomically rename.

## 7. Evaluation and claims protocol

### Fixed suites and metrics

Prepare each suite before Stage 1 and store an immutable manifest, prompt-format version, and hash.

| Suite | What it measures |
| --- | --- |
| In-distribution by family and difficulty | Capability on known templates with unseen seeds. |
| Held-out template families | Structural generalization to unseen compositions. |
| Numeric extrapolation | Behavior outside training number ranges. |
| Format and parser stress | Output validity and verifier robustness. |
| General-text validation | Forgetting during math specialization. |
| Held-out preference pairs | DPO discrimination without prompt overlap. |

For every model branch, report exact-answer accuracy, valid-format rate, pass@1, empirical pass@4, accuracy by family and difficulty, extrapolation accuracy, mean and distribution of generated tokens, validation cross-entropy, and inference latency under a stated device. Report train tokens, examples, optimizer steps, GPU-hours, and estimated spend. For GRPO, also report mixed-reward group fraction and verifier-error rate.

Use deterministic greedy decoding for the primary exact-answer accuracy. Run a separate fixed-seed stochastic evaluation for sampled pass@1 and pass@4 with four actual samples per prompt, the same temperature and length cap for every checkpoint. Sampled pass@1 is the mean correctness of individual samples; pass@4 is the fraction of prompts with at least one correct sample. Do not estimate pass@4 as 1 − (1 − pass@1)^4; generations are correlated. Use identical prompts, tokenizer, answer parser, and sampling settings across branches. Record failures as wrong, invalid-format, truncated, or verifier-error; do not drop them from the denominator.

Compute Wilson intervals for individual accuracy and paired bootstrap intervals for checkpoint differences using prompt IDs as resampling units. Predeclare the primary comparison and test suite before final evaluation. If multiple families or hyperparameters are searched, label those results exploratory and keep the final held-out comparison distinct. Report numerator, denominator, interval method, and bootstrap seed. When practical, rerun the decisive comparison with at least two training seeds. A single seed can support an exploratory finding but not a broad claim of reliable improvement.

The evaluation harness should produce one machine-readable row per prompt and checkpoint, plus an aggregate table. The row includes prompt ID, suite hash, checkpoint hash, decode config, raw completion, parsed answer, exact expected answer, verifier result, token count, and latency. Keep raw completions available for error audit but exclude private licensed corpus text from public reports.

### Acceptance criteria

These are **proposed gates**, not predicted model performance. Do not change frozen tests to satisfy them.

| Gate | Acceptance condition |
| --- | --- |
| Engineering readiness | Stage 0 deterministic generation, causality, loss masking, verifier, checkpoint, and resume tests pass; one tiny batch overfits. |
| Data readiness | Every held-out manifest is frozen and cross-split collision checks pass. Source licenses are recorded. |
| Stage completion | Each run logs actual token and step counts, saves a loadable checkpoint, and completes the identical evaluation suite. |
| SFT readiness for RL | In an RL-pool pilot, some responses are correct and a useful fraction of groups mix correct and wrong outcomes. If not, revise training-only tasks or SFT. |
| Research success for H4 | On the predeclared held-out-template suite, v4-grpo exceeds both v2-sft and the matched-cost v2-extra-sft by at least 2 percentage points, and the paired 95% interval for each difference excludes zero. Treat this threshold as a decision rule, not a guarantee. |
| Honest closeout | Publish or save the report even if H4 fails, if budget ends, or if the test is inconclusive. State limitations and any protocol deviations. |

An aspirational diagnostic target is 95% valid-answer formatting and 80% accuracy on the easiest arithmetic suite after SFT. These numbers were not measured and are not pass/fail rules for scientific validity. If the model cannot meet them under the budget, report that result and choose whether to stop or propose a separately authorized revision.

### Failure and validity checks

Before trusting a gain, inspect leakage, duplicate prompts, decoder settings, parser false positives, changed response lengths, different prompt instructions, and selection on the final test. Compare both exact accuracy and format, because a format-only gain can inflate a metric if the parser changes. For a negative GRPO result, inspect baseline success, mixed-group frequency, reward implementation, clip fraction, KL drift, generation truncation, and compute mismatch. Do not explain a result with one favored cause unless an ablation isolates it.

Report look-ahead and leakage risk for all training-data decisions. If any external benchmark is used, document its likely pretraining contamination risk and do not claim a clean generalization result from it. Report sensitivity to seed, template distribution, number range, training-token budget, and compute. No trading or financial performance claims belong in this project.

## 8. Compute budget and authorization

The intended first GPU is one 24 GB RTX 4090. A CPU or Mac can develop generators, tests, and tiny model runs. A 24 GB card is a plausible choice for this approximately 43M-parameter model, but full-sequence training and GRPO memory must pass an actual smoke test. An A100 80 GB is an optional comfort choice only after measured need and a new cost approval. Do not assume that a rented GPU has the same effective throughput as a local benchmark.

As checked on **7 November 2026**, [RunPod's GPU listing](https://www.runpod.io/gpu-models) showed an RTX 4090 Community Cloud listing at **$0.34 per hour**, and [RunPod's pricing page](https://www.runpod.io/pricing) showed an RTX 4090 Secure Cloud listing at **$0.89 per hour**. These are listed rates, not a quote for an available pod. Availability, storage, transfer, idle pod time, taxes, and pricing may change. Recheck the actual checkout price and billing model immediately before any user-authorized rental.

| Illustrative GPU use | At $0.34/hour | At $0.89/hour |
| ---: | ---: | ---: |
| 25 hours | $8.50 | $22.25 |
| 50 hours | $17.00 | $44.50 |
| 100 hours | $34.00 | $89.00 |

This table excludes all non-GPU charges. The earlier conversation suggested a roughly **$50 initial cap**. Treat it as a proposed first authorization ceiling, not available funds. A possible later $100 total project ceiling is a planning scenario only. The default allowed spend is **$0** until the user explicitly approves a provider, total dollar limit, time limit, and termination rule. Do not ask for open-ended permission. Do not provision a pod from this specification alone.

For sizing only, the common 6 × parameter count × training tokens approximation gives about 41 petaflop of weight-matrix training work for 42.74M parameters and 160M base-plus-mid tokens. This omits attention, optimizer, data loading, evaluation, rollouts, DPO reference inference, and GRPO generation. It is **not a runtime forecast**. Measure tokens per second, GPU utilization, peak memory, and billed rate in a short authorized pilot. Then estimate each stage as remaining processed tokens divided by observed throughput, plus measured stage overhead and a contingency. If the projected bill exceeds the approved limit, stop before starting the stage or reduce the planned token budget with a recorded protocol change.

Every paid run requires a machine-readable budget ledger with approved maximum, committed spend, estimated accrued spend, projected next-stage spend, provider rate, pod ID, and shutdown time. The runner must stop on either the time or dollar ceiling and save a resumable checkpoint before shutdown where possible. Stop or terminate idle compute according to the provider's actual billing rules. Never leave a pod running just to preserve state; upload or copy checkpoints to an approved destination first if storage cost and privacy permit. Do not create provider accounts, install paid services, or access credentials on the user's behalf without authorization.

## 9. Repository, interfaces, and reproducibility

Create a small Python repository in the existing MicroReason folder. Use the project's chosen lockfile and pin package versions after the first local install. Python 3.11 or later and PyTorch are proposed. Implement the custom decoder as a Hugging Face Transformers-compatible local model and config if that can be done without changing the architecture; this permits standard save/load and may ease TRL integration. Prove adapter compatibility in Stage 0. [TRL's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes custom reward functions, but its current API must be pinned and tested in this project. If a compatible adapter becomes costly or distorts the baseline, implement the minimal DPO and GRPO loops in PyTorch, verify their math on tiny cases, and document that decision. Do not silently replace GRPO with another algorithm.

Proposed structure:

~~~text
MicroReason/
├── MICROREASON_MASTER_SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock or another chosen lockfile
├── configs/
│   ├── model_43m.yaml
│   ├── smoke.yaml
│   ├── pretrain.yaml
│   ├── midtrain.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/microreason/
│   ├── cli.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── checkpoint.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── generators.py
│   │   ├── split.py
│   │   └── manifests.py
│   ├── verify/
│   │   ├── parser.py
│   │   ├── exact.py
│   │   └── symbolic.py
│   ├── train/
│   │   ├── common.py
│   │   ├── pretrain.py
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   └── grpo.py
│   ├── eval/
│   │   ├── generate.py
│   │   ├── score.py
│   │   └── report.py
│   └── serve.py
├── tests/
│   ├── test_generators.py
│   ├── test_verifier.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_resume.py
│   └── test_training_objectives.py
├── data/manifests/
├── artifacts/
├── runs/
└── reports/
~~~

Keep large datasets, raw licensed corpora, provider credentials, and checkpoint binaries out of Git. Track manifests, configs, hashes, code, aggregate metrics, and a model card. Use a local artifact store with documented paths and checksums. A public release requires a separate review of source licenses and model outputs.

Implement these prospective CLI entry points. The exact flags may change during Stage 0, but each operation must remain scriptable and noninteractive.

~~~text
microreason data generate --config configs/data.yaml --out data/manifests/
microreason data audit --manifest data/manifests/train_v1.json
microreason tokenizer train --config configs/tokenizer.yaml
microreason model inspect --config configs/model_43m.yaml
microreason train --config configs/pretrain.yaml
microreason train --config configs/midtrain.yaml --parent artifacts/v0-base
microreason train --config configs/sft.yaml --parent artifacts/v1-math
microreason train --config configs/dpo.yaml --parent artifacts/v2-sft
microreason train --config configs/grpo.yaml --parent artifacts/v2-sft
microreason eval --suite data/manifests/test_v1.json --checkpoint artifacts/v2-sft
microreason compare --runs runs/experiment_v1/
microreason serve --checkpoint artifacts/v4-grpo --host 127.0.0.1
~~~

The CLI must print the resolved config, run ID, parent, data hashes, and projected spend before training. The paid-run path must fail if authorization fields are absent or the estimated limit is exhausted. Local serving binds to loopback by default. Quantization is optional after the model and evaluator agree on outputs; never substitute a quantized checkpoint for the primary comparisons.

## 10. Implementation phases for the next agent

Work in small, independently checked phases. Read any AGENTS.md, CLAUDE.md, README, and nearby code before changing an existing repository. The folder was empty when this brief was created, but inspect it again because the user may add files.

1. **Initialize the repository.** Add package metadata, a README that states the unimplemented status, typed config schema, logging, a test runner, and a lockfile. Verify local installation and CLI help. Do not start GPU training.
2. **Build the data and verifier first.** Implement exact-arithmetic generators, template-family splits, manifest hashing, deduplication, restricted answer parsing, and independent checks. Run property tests and adversarial verifier tests. Freeze evaluation manifests only after reviewing their distributions and overlap report.
3. **Build tokenizer and model.** Train the BPE tokenizer on training-only data. Implement the full and smoke configurations, count parameters, test causality and loss masking, and verify save/load. Record the exact token format.
4. **Build and prove the common training loop.** Overfit one batch, test accumulation against an equivalent large batch where numerically feasible, test checkpoint resume, and measure memory on the smoke model. Keep a short CPU or MPS run repeatable.
5. **Request a bounded GPU pilot only after local gates pass.** Present the user with the exact provider, GPU, listed rate, maximum dollars, maximum runtime, shutdown rule, and planned smoke tests. Await explicit approval. Once approved, run only within those bounds and report measured throughput and memory.
6. **Run base and mid-training within an approved budget.** Save v0-base and v1-math, record processed tokens, run the frozen validation suites, and stop if loss diverges or budget forecasts exceed the limit.
7. **Run SFT and branch experiments.** Save v2-sft. Check RL baseline success. Create DPO pairs from training prompts, run the DPO pilot and branch. Prepare the equal-compute SFT control. Run GRPO only if the reward-variance gate passes and the remaining budget permits it.
8. **Evaluate and report.** Run identical frozen suites on every checkpoint. Produce per-example and aggregate tables, paired intervals, cost ledger, failure taxonomy, model card, and a technical report. State whether each hypothesis was supported, contradicted, or inconclusive. Ask for publication approval only after a reviewable draft exists.

Milestone artifacts are, in order, a local green test report, a frozen data-manifest bundle, a tokenizer and model inspection report, an authorized GPU pilot report, stage checkpoint manifests, and a final comparison report. Each phase ends with a check against its actual artifact before the next begins.

## 11. Safety, security, and reporting limits

Treat generated data and model completions as untrusted text. Parse with a restricted grammar and resource limits. Do not execute completions, load arbitrary serialized Python objects from untrusted checkpoints, or send private source text to a third-party API. Keep credentials in an approved secret store or environment mechanism, never in configs, logs, notebooks, or reports. Never print secret values. Make all artifact imports check hashes and expected formats.

Use only sources whose training and redistribution terms are documented. Do not assume that a publicly accessible dataset is licensed for training or release. The model is for research on arithmetic and algebra, not medical, legal, or financial advice. A local API may be useful for demonstration, but it must state the narrow domain and can fail outside it. No public endpoint, model upload, repo push, or deployment follows automatically from finishing training.

The final report must separate:

- **Implemented and tested behavior**, with commands, code revision, artifact hashes, and test results.
- **Measured experimental results**, with counts, uncertainty, seeds, and actual billed cost when available.
- **Estimates and unrun proposals**, including future training budgets and ablations.
- **Known limitations**, including synthetic-data bias, template leakage risk, number-range sensitivity, small-model capacity, incomplete seed replication, and possibly unfaithful generated explanations.

Do not describe an exploratory single run as a general real-world result. Preserve negative outcomes, stopped runs, and failed gates in the run ledger.

## 12. Agent completion checklist

The implementing agent may call the project complete only when it can point to all applicable evidence below. A budget-limited partial project must say exactly which stages remain unrun.

- The architecture parameters and instantiated count match the recorded config.
- Generator, split, tokenizer, parser, exact verifier, model, loss, and resume tests pass.
- Test manifests are frozen and their hashes and overlap audits are recorded.
- Every executed stage has a loadable, immutable checkpoint and a run manifest.
- v2-sft, v2-extra-sft, v3-dpo, and v4-grpo are clearly identified as completed, skipped, stopped, or blocked. No missing branch is implied to exist.
- The primary comparison uses the same prompts, tokenizer, verifier, and decoding protocol, with uncertainty and cost reported.
- The cost ledger stays within the user's latest explicit authorization.
- The report states supported, unsupported, or inconclusive findings for H1 through H4, with no unmeasured performance or deployment claim.

This specification authorizes local planning and implementation work only. Any GPU spend or external release requires the user's separate approval.


