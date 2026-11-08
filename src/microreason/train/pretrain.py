"""Build a causal-LM run (smoke, pretrain, midtrain) from a resolved stage config."""

from __future__ import annotations

from pathlib import Path

import torch

from microreason.config import ResolvedConfig
from microreason.data.tokenized import TokenizedDataset
from microreason.tokenizer import MicroTokenizer
from microreason.train.common import Trainer
from microreason.train.data import PackedStream, Source, fixed_eval_batches

CAUSAL_LM_STAGES = ("smoke", "pretrain", "midtrain")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset(path: str, tok: MicroTokenizer, expected_split: str) -> TokenizedDataset:
    ds = TokenizedDataset(path, verify=True)
    if ds.manifest.get("tokenizer_sha256") != tok.sha256:
        raise ValueError(f"{path} was tokenized with a different tokenizer")
    if ds.manifest.get("split") != expected_split:
        raise ValueError(f"{path} has split '{ds.manifest.get('split')}', need '{expected_split}'")
    return ds


def build_trainer(resolved: ResolvedConfig, run_dir: str | Path, device: str) -> Trainer:
    stage = resolved.stage
    if stage.stage not in CAUSAL_LM_STAGES:
        raise NotImplementedError(f"stage '{stage.stage}' is not a causal-LM stage")
    if not stage.mixture:
        raise ValueError("causal-LM stages need a data mixture")
    if stage.tokenizer is None:
        raise ValueError("stage config must name a tokenizer")
    tok = MicroTokenizer.load(stage.tokenizer)
    if tok.vocab_size != resolved.model.vocab_size:
        raise ValueError("tokenizer vocab size does not match the model config")

    sources = [
        Source(Path(m.dataset).name, _dataset(m.dataset, tok, "train"), m.weight)
        for m in stage.mixture
    ]
    stream = PackedStream(sources, stage.sequence_length, stage.seed, tok.bos_id, tok.eos_id)
    micro = stage.micro_batch_size or stage.sequences_per_step
    eval_sets = {}
    for v in stage.validation:
        ds = _dataset(v.dataset, tok, "val")
        eval_sets[ds.name] = fixed_eval_batches(
            ds, v.sequences, stage.sequence_length, micro, tok.bos_id, tok.eos_id
        )

    model, parent_manifest = None, None
    if stage.parent:
        from microreason.checkpoint import load_checkpoint

        model, _, parent_manifest = load_checkpoint(stage.parent, "cpu")

    provenance = {
        "tokenizer_sha256": tok.sha256,
        "train_datasets": {s.name: s.dataset.fingerprint for s in sources},
        "validation_datasets": {
            Path(v.dataset).name: TokenizedDataset(v.dataset).fingerprint for v in stage.validation
        },
        "parent_config_hash": parent_manifest["config_hash"] if parent_manifest else None,
        "parent_step": parent_manifest["step"] if parent_manifest else None,
    }
    return Trainer(resolved, run_dir, stream, eval_sets, device, provenance, model)
