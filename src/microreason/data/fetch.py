"""Download pinned source files and verify their sha256 before they are used.

Files are streamed to a temporary path, hashed, and atomically renamed only if the hash matches.
An existing file with the correct hash is skipped; one with a wrong hash is an error.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from microreason.data.sources import SourceFile, TextSource
from microreason.logging_utils import get_logger

log = get_logger("microreason.data.fetch")

_CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    pass


def file_url(source: TextSource, f: SourceFile) -> str:
    return f"https://huggingface.co/datasets/{source.hub_repo}/resolve/{source.revision}/{f.path}"


def local_path(root: Path, source: TextSource, f: SourceFile) -> Path:
    return root / source.name / source.revision / f.path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch_file(source: TextSource, f: SourceFile, root: Path) -> Path:
    dest = local_path(root, source, f)
    if dest.exists():
        actual = sha256_file(dest)
        if actual != f.sha256:
            raise ChecksumError(f"{dest} exists with wrong sha256 {actual}; remove it and retry")
        log.info("ok (cached) %s", f.path)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    url = file_url(source, f)
    log.info("downloading %s (%.0f MB)", f.path, f.size_bytes / 1e6)
    h = hashlib.sha256()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "microreason-fetch/0.0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            h.update(chunk)
            size += len(chunk)

    if size != f.size_bytes or h.hexdigest() != f.sha256:
        tmp.unlink(missing_ok=True)
        raise ChecksumError(
            f"{f.path}: got {size} bytes sha256={h.hexdigest()}, "
            f"expected {f.size_bytes} bytes sha256={f.sha256}"
        )
    os.replace(tmp, dest)
    log.info("ok %s", f.path)
    return dest


def fetch_source(source: TextSource, root: Path) -> list[Path]:
    return [fetch_file(source, f, root) for f in source.files]
