import hashlib
import io

import pytest

from microreason.data import fetch
from microreason.data.sources import SourceFile, TextSource

PAYLOAD = b"hello cosmopedia"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
PAYLOAD_SIZE = len(PAYLOAD)


def _source(sha=PAYLOAD_SHA, size=PAYLOAD_SIZE):
    return TextSource(
        name="t",
        hub_repo="org/ds",
        revision="a" * 40,
        license="mit",
        license_url="u",
        retrieved_metadata_on="2026-11-07",
        provenance="p",
        files=[SourceFile(path="data/x/f.parquet", size_bytes=size, sha256=sha)],
    )


@pytest.fixture
def fake_net(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    return calls


def test_url_pins_revision():
    s = _source()
    assert fetch.file_url(s, s.files[0]).endswith(f"/resolve/{'a' * 40}/data/x/f.parquet")


def test_fetch_verifies_and_caches(tmp_path, fake_net):
    s = _source()
    (p,) = fetch.fetch_source(s, tmp_path)
    assert p.read_bytes() == PAYLOAD
    fetch.fetch_source(s, tmp_path)
    assert len(fake_net) == 1  # second call used the verified cached file


def test_bad_checksum_leaves_no_file(tmp_path, fake_net):
    s = _source(sha="b" * 64)
    with pytest.raises(fetch.ChecksumError):
        fetch.fetch_source(s, tmp_path)
    assert not any(tmp_path.rglob("*.parquet*"))
