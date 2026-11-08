"""Atomic, checksummed, never-overwritten training checkpoints (spec §2 invariants 3-4, §6).

Layout: <run_dir>/checkpoints/step_0000500/
    model/                 save_pretrained output (safetensors + config.json)
    training_state.pt      optimizer, step, consumed tokens, data stream state, RNG states
    manifest.json          provenance and sha256 of every file above

A checkpoint is written to a temporary sibling directory, its files are checksummed and
re-verified, and only then is the directory renamed into place. Loading verifies every checksum
and uses torch.load(weights_only=True), so no arbitrary pickled objects are executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

MANIFEST = "manifest.json"
STATE = "training_state.pt"
MODEL_DIR = "model"


class CheckpointError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }


def checkpoint_dir(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step_{step:07d}"


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model,
    optimizer: torch.optim.Optimizer,
    state: dict,
    provenance: dict,
) -> Path:
    final = checkpoint_dir(run_dir, step)
    if final.exists():
        raise CheckpointError(f"{final} already exists; checkpoints are never overwritten")
    tmp = final.with_name(f".tmp_{final.name}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        model.save_pretrained(tmp / MODEL_DIR)
        torch.save({"optimizer": optimizer.state_dict(), **state}, tmp / STATE)
        files = _file_hashes(tmp)
        manifest = {"schema_version": 1, "step": step, "files": files, **provenance}
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if _file_hashes(tmp) != files:
            raise CheckpointError("checkpoint files changed while being verified")
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_json_atomic(Path(run_dir) / "latest.json", {"step": step, "path": str(final)})
    return final


def verify_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    mpath = path / MANIFEST
    if not mpath.exists():
        raise CheckpointError(f"{path} has no manifest")
    manifest = json.loads(mpath.read_text())
    actual = _file_hashes(path)
    if actual != manifest["files"]:
        bad = sorted(set(actual.items()) ^ set(manifest["files"].items()))
        raise CheckpointError(f"{path} failed checksum verification: {bad[:3]}")
    return manifest


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Return (model, training_state, manifest) after verifying every checksum."""
    from microreason.model import MicroReasonForCausalLM

    path = Path(path)
    manifest = verify_checkpoint(path)
    model = MicroReasonForCausalLM.from_pretrained(path / MODEL_DIR).to(device)
    state = torch.load(path / STATE, map_location="cpu", weights_only=True)
    return model, state, manifest


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    p = Path(run_dir) / "latest.json"
    return Path(json.loads(p.read_text())["path"]) if p.exists() else None
