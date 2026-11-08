"""Compare rebuilt data against the fingerprints recorded on the reference machine.

    python scripts/check_fingerprints.py           # verify (exit 1 on any mismatch)
    python scripts/check_fingerprints.py --write   # record current fingerprints (reference machine)

Covers the tokenizer, the general-text split, every math split and every tokenized dataset, so a
pod run is known to train on byte-identical inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED = Path("scripts/expected_fingerprints.json")


def current() -> dict:
    fp: dict = {}
    tok = json.loads(Path("artifacts/tokenizer/v1/tokenizer_manifest.json").read_text())
    fp["tokenizer"] = tok["tokenizer_sha256"]
    split = Path("/tmp/general_text_split_v1.json")
    if not split.exists():
        split = Path("data/manifests/general_text_split_v1.json")
    fp["general_text_val_digest"] = json.loads(split.read_text())["val_digest_set_sha256"]
    for m in sorted(Path("data/manifests").glob("math_*_v1.json")):
        if "frozen" in m.name:
            continue
        fp[f"math:{m.stem}"] = json.loads(m.read_text())["sha256"]
    for d in sorted(Path("data/tokenized").glob("*_v1")):
        man = json.loads((d / "manifest.json").read_text())
        fp[f"tokenized:{d.name}"] = {"tokens": man["tokens"], "files": man["files"]}
    return fp


def main() -> int:
    fp = current()
    if "--write" in sys.argv:
        EXPECTED.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
        print(f"wrote {EXPECTED} ({len(fp)} entries)")
        return 0
    want = json.loads(EXPECTED.read_text())
    if "--skip-stream" in sys.argv:  # stream v1 is not rebuilt on scaled (v2) pods
        want = {k: v for k, v in want.items() if "stream" not in k}
        fp = {k: v for k, v in fp.items() if "stream" not in k}
    if "--only" in sys.argv:
        prefix = sys.argv[sys.argv.index("--only") + 1] + ":"
        keep = {k for k in want if k.startswith(prefix) and k != "math:math_train_stream_v1"}
        want = {k: want[k] for k in keep}
        fp = {k: fp.get(k) for k in keep}
    bad = [k for k in sorted(set(want) | set(fp)) if want.get(k) != fp.get(k)]
    for k in bad:
        print(f"MISMATCH {k}: expected {want.get(k)} got {fp.get(k)}")
    print("fingerprints match" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
