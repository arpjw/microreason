# MicroReason

Small decoder-only language models (42.7 M parameters) trained **from random initialization** to
do exact arithmetic and linear algebra with step-by-step work. The project asked whether a staged
pipeline can teach a tiny model to reason within a small compute budget. The pipeline is base
pretraining → math mid-training → SFT → RL. Every hypothesis was pre-registered and tested once
on sealed test suites, for a total GPU spend of about $23.

- **Models:** [huggingface.co/arpjw/microreason-43m](https://huggingface.co/arpjw/microreason-43m)
- **Final report:** [`reports/closeout.md`](reports/closeout.md)
- **Design spec:** [`MICROREASON_MASTER_SPEC.md`](MICROREASON_MASTER_SPEC.md)

**Status:** closed (2026-11-09).

## Results

The final models come from protocol v4:

| Model | Held-out-shape test | In-distribution test | 4–5 digit extrapolation |
| --- | ---: | ---: | ---: |
| `sft-long`: SFT model (3.1 M supervised examples) | 81.4% | 87.1% | 10.2% |
| `grpo`: `sft-long` + GRPO (150 on-policy steps, outcome reward) | 82.0% | 87.8% | 10.1% |
| `extra-sft`: `sft-long` + continued SFT for the same GPU time as GRPO | 83.1% | 88.8% | 7.8% |

**How these numbers were measured:**
- **Decoding:** greedy, show-work format, exact final answer.
- **Suite sizes:** 6,000 / 7,000 / 7,000 problems.
- **Sealing:** the suites were frozen before training and opened once.
- **Held out:** a held-out shape is an expression structure never seen in training.

**Pre-registered hypotheses** (spec §7):

| ID | Hypothesis | Finding | Evidence |
| --- | --- | --- | --- |
| H1 | Math mid-training improves math capability | **Supported** | +5.29 pts [+4.49, +6.14] on `test_id` |
| H2 | SFT gives instruction following and answer format | **Supported** | 0.0% → 79.5% under SFT prompts on `test_id` |
| H3 | DPO improves format without a major correctness loss | **Not supported** | format already 100% valid; −5.1 pts on `test_id` |
| H4 | Outcome-reward GRPO beats SFT and equal-cost continued SFT on held-out problems | **Not supported** | +0.49 [+0.09, +0.98] vs parent; −1.06 [−1.76, −0.53] vs continued SFT |

**Confirmed secondary finding.** RL *preserves* the ability to handle numbers larger than any seen
in training, which continued SFT erodes.
- **First result (v4):** GRPO beat time-matched continued SFT by +2.18 points
  [+0.56, +3.88].
- **Replication:** +2.17 [+0.62, +3.78] on a fresh, separately frozen suite.
- **Consistency:** all three seeds were positive both times.

## What the models do

**Expressions:** expressions with 1–5 operators over integers up to 999 (`+ − × ÷`, parentheses),
with exact rational results.

**Equations:** linear equations such as `3(x + 4) − 2 = 5x − 10`.

**Output format:** each model writes numbered reduction steps, then an answer:

```
[steps]
1. Expand: 3x + 12 - 2 = 5x - 10
2. Combine constants: 3x + 10 = 5x - 10
3. Subtract 5x from both sides: -2x + 10 = -10
4. Subtract 10 from both sides: -2x = -20
5. Divide both sides by -2: x = 10
[/steps]
[answer]10[/answer][eos]
```

**Accuracy falls with depth:** about 98% with 1 operator, 76% with 3 and 60% with 5. Linear
equations are about 99%.

## Usage

**From Hugging Face:** no checkout needed; the model code ships with the weights.

```python
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import snapshot_download

path = snapshot_download("arpjw/microreason-43m", allow_patterns=["grpo/*"]) + "/grpo"
tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).eval()

prompt = "[bos][user]Compute: (12 - 5) * 3 + 4 / 2 Show your work.[eos][assistant]"
ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]])
eos = tok.convert_tokens_to_ids("[eos]")
with torch.no_grad():
    for _ in range(320):
        nxt = model(ids).logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == eos:
            break
print(tok.decode(ids[0]))
```

**From this repository:**
- **Requires** [uv](https://docs.astral.sh/uv/); Python 3.13 is pinned.
- **Evaluation is protected:** it only runs on frozen suites, and test suites refuse to open
  without `--allow-test`.

```bash
uv sync
uv run pytest
uv run microreason eval --suite val --checkpoint <checkpoint> --format sft-show-work-v1 \
    --data-version v3 --tokenizer artifacts/tokenizer/v2 --kv-cache --max-new-tokens 320 \
    --out runs/eval/<name>
uv run microreason compare --a runs/eval/<a> --b runs/eval/<b>    # paired bootstrap, b − a
```

**Prompt formats** (the markers are single tokens):

| Format | Prompt | Used for |
| --- | --- | --- |
| `sft-show-work-v1` | `[bos][user]{problem} Show your work.[eos][assistant]` | SFT and RL models |
| `sft-direct-v1` | `[bos][user]{problem} Give only the answer.[eos][assistant]` | SFT models; about 35% accuracy |
| `plain-v1` | `[bos]{problem}\n` | base and mid-trained models; they continue to `Answer: x` |

## Training

| Stage | Data | Notes |
| --- | --- | --- |
| Base pretraining | 1.0 B tokens: 70% general text ([Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)), 30% synthetic math | from random init |
| Math mid-training | 200 M tokens, 85% math with an easy → hard curriculum | |
| SFT | 100 k, then 3 M more verified worked solutions (75% show-work, 25% answer-only) | `sft-long` |
| GRPO | 150 steps × 1,024 completions, group 16, binary exact-answer reward, pass-rate filter (2–14 of 16 correct), LR 3e-6, KL 1e-3, one update per batch | `grpo` |
| Continued SFT | about 2.2 M fresh worked solutions in the same 46 min of GPU time | `extra-sft` |

**Model:** 12 layers × 512 dimensions, 8 heads, RMSNorm, RoPE, SwiGLU, tied embeddings.

**Tokenizer:** an 8,192-token byte-level BPE that splits every number into single digits.

**Math data:** generated by a seeded expression-tree grammar; every training problem is checked
against the frozen evaluation suites to exclude overlap.

## What we learned

1. **Data representation was the bottleneck, not the training method.**
   - **Before:** with ordinary BPE numbers and 14 fixed templates, every model scored ≤ 0.2% on
     held-out templates.
   - **After:** digit-level tokens plus tree-generated problems reached 70% on held-out shapes,
     with the same model and budget.
2. **GRPO needs the published recipe, even at 43 M parameters.**
   - **Before:** with off-policy minibatches, std-normalised advantages, a 10× higher LR and a
     strong KL, GRPO lost to its parent.
   - **After:** with on-policy updates, Dr. GRPO-style normalisation and a pass-rate-filtered
     prompt stream, it gains on every seed.
3. **With unlimited correct worked solutions, extra SFT beats RL at equal compute.** In GRPO's
   46 minutes, continued SFT processed about 2.2 M fresh gold traces; GRPO trained on about 154 k
   of its own samples.
4. **RL's advantage here is protection, not improvement.** Extra SFT erodes extrapolation to
   larger numbers; GRPO and RFT don't.
5. **GRPO's negative term added nothing over rejection-sampling fine-tuning** (+0.13, interval
   includes 0). A bounded step-level process reward did not help either.

## History

| Protocol | What changed | What it found | Spend | Report |
| --- | --- | --- | ---: | --- |
| v1 | 14 fixed templates, BPE tokenizer | Pipeline works; mid-training +23 pts on val | ≈ $1.1 | [`stage1_v1`](reports/stage1_v1.md), [`stage2_v1`](reports/stage2_v1.md) |
| v2 | 1 B base + 200 M mid tokens; GRPO, DPO, equal-compute SFT | ≤ 0.2% on held-out templates for every model; GRPO and DPO lost | ≈ $7.3 | [`closeout_v1`](reports/closeout_v1.md), [`failure_analysis_v1`](reports/failure_analysis_v1.md) |
| v3 | Digit tokenizer, tree-generated problems, held-out shapes | 70.3% on held-out shapes; the RL gate failed twice | ≈ $3.4 | [`v3_phase_b`](reports/v3_phase_b.md), [`protocol_v3`](reports/protocol_v3.md) |
| v4 | Literature review; much longer SFT; filtered prompt stream; published GRPO recipe; 3 seeds | GRPO +0.5 vs parent, −1.1 vs continued SFT | ≈ $10.5 | [`v4_results`](reports/v4_results.md), [`protocol_v4`](reports/protocol_v4.md), [review](<reports/GRPO gains for tiny models.md>) |
| v4x | Evaluation only, on a fresh frozen extrapolation suite | Extrapolation finding replicated | ≈ $1.0 | [`replication_v4x`](reports/replication_v4x.md) |

## Repository layout

| Path | Contents |
| --- | --- |
| `src/microreason/model.py` | The model (Hugging Face `PreTrainedModel`, KV cache) |
| `src/microreason/data/` | Math generators (v1 templates, v3 expression trees), splits, manifests, training pools |
| `src/microreason/verify/` | Exact answer and step verifiers |
| `src/microreason/train/` | Training loop, SFT, budget guard |
| `src/microreason/rl/` | GRPO / RFT (`grpo.py`), DPO, rollouts, gates |
| `src/microreason/eval/` | Generation, scoring, paired and cluster bootstrap |
| `configs/` | Model, data, stage, RL and budget configs |
| `scripts/` | Pod setup and session scripts, decision rules |
| `data/manifests/` | Hashes and counts of every dataset; frozen evaluation suites (v1, v3, v4x) |
| `reports/` | Protocols, confirmatory plans, results, closeout |
| `tests/` | About 350 tests |

Generated data (`data/generated/`), runs and checkpoints (`runs/`) are not tracked. The data is
rebuilt deterministically from seeds, and each rebuild must match the committed hashes.

## Reproducing

Heavy work runs on rented GPU pods; the Mac only runs tests and small checks. The v4 pipeline,
on a RunPod pod with this repository pushed by `scripts/pod_push.sh`:

```bash
SETUP_PROFILE=v4 scripts/pod_setup.sh    # rebuild and verify frozen suites, build the v4 SFT pool
scripts/pod_v4_phase_b.sh                # v2-sft-long, gates, LR pilots (held-out-val only)
SETUP_PROFILE=v4c scripts/pod_setup.sh   # rebuild the RL pool, check it matches Phase B
scripts/pod_v4_phase_c.sh                # 3 GRPO seeds, time-matched controls, the single test look
```

Earlier protocols have their own `scripts/pod_*.sh` and configs. See
[`scripts/PILOT_RUNBOOK.md`](scripts/PILOT_RUNBOOK.md) for pod setup, the self-terminating
watchdog and collecting results.

**Spending safeguards:**
- **Off by default:** configs default to `paid_compute_allowed: false` and `max_spend_usd: 0`.
  A paid run needs a provider, a dollar cap, a runtime cap and a shutdown rule in its config, plus
  explicit owner approval.
- **Budget guard:** it stops training before either ceiling and saves a checkpoint.
- **Watchdog:** a watchdog on the pod deletes the pod at a hard deadline.

## Limitations

- Narrow domain: arithmetic and linear equations only, in the training phrasing. These are not
  general chat or general-purpose math models.
- Accuracy on numbers beyond 3 digits is low (about 10%).
- Every test suite in this repository has now been used. Any further claim needs freshly
  generated, separately frozen suites (`microreason data generate-replication-v4x` shows how).
- Small models trained on a narrow distribution; outputs should not be relied on without
  checking.
