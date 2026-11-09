"""scripts/v4_decide.py: cluster bootstrap and the pre-declared protocol v4 rules on synthetic
eval directories."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v4_decide", ROOT / "scripts" / "v4_decide.py")
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)

SW = vd.SW


def _write(ev: Path, name: str, suite: str, correct: list[bool], fmt: str = SW, shapes=40,
           summary: dict | None = None) -> None:  # fmt: skip
    d = ev / f"{name}_{suite}_{fmt}"
    d.mkdir(parents=True)
    rows = [{"prompt_id": f"p{i:05d}", "template_id": f"s{i % shapes}", "decode": "greedy",
             "status": "correct" if c else "wrong"} for i, c in enumerate(correct)]  # fmt: skip
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    acc = sum(correct) / len(correct)
    s = {"greedy": {"exact_accuracy": {"value": acc, "numerator": sum(correct)}}}
    (d / "summary.json").write_text(json.dumps({**s, **(summary or {})}))


def _flip(base: list[bool], k: int, seed: int) -> list[bool]:
    """Make k wrong answers right (a gain of k / n)."""
    rng = random.Random(seed)
    out = list(base)
    wrong = [i for i, c in enumerate(out) if not c]
    for i in rng.sample(wrong, k):
        out[i] = True
    return out


def _base(n=4000, p=0.6, seed=0):
    rng = random.Random(seed)
    return [rng.random() < p for _ in range(n)]


def test_cluster_bootstrap_point_and_interval():
    a = {f"p{i}": (f"s{i % 50}", c) for i, c in enumerate(_base())}
    b = {k: (s, True if int(k[1:]) % 10 == 0 else c) for k, (s, c) in a.items()}
    r = vd.cluster_bootstrap([(a, b)], resamples=2000)
    gain = sum(b[k][1] for k in a) / len(a) - sum(a[k][1] for k in a) / len(a)
    assert abs(r["difference"] - gain) < 1e-12 and r["interval_above_zero"]
    assert r["bootstrap95"][0] < gain < r["bootstrap95"][1] and r["clusters"] == 50
    assert vd.cluster_bootstrap([(a, a)], resamples=500)["difference"] == 0
    with pytest.raises(ValueError):
        vd.cluster_bootstrap([(a, dict(list(b.items())[:-1]))])


def _arms(ev: Path, gains: dict[str, int], n=4000):
    base = _base(n)
    _write(ev, "v2-sft-long", "test_heldout_shape", base)
    for name, k in gains.items():
        _write(ev, name, "test_heldout_shape", _flip(base, k, hash(name) % 1000))


def test_h4_supported_only_when_every_part_of_the_rule_holds(tmp_path):
    n = 4000
    big = int(0.04 * n)
    ev = tmp_path / "ok"
    _arms(ev, {**{f"v4-grpo-s{s}": big for s in (1, 2, 3)},
               **{f"v2-extra-sft-s{s}": 10 for s in (1, 2, 3)}})  # fmt: skip
    vd.main(["h4", "--eval-dir", str(ev), "--out", str(tmp_path / "ok.json")])
    out = json.loads((tmp_path / "ok.json").read_text())
    assert out["H4"]["status"] == "supported" and out["H1"]["status"] == "undecided"
    # control as good as GRPO -> not supported
    ev2 = tmp_path / "ctrl"
    _arms(ev2, {**{f"v4-grpo-s{s}": big for s in (1, 2, 3)},
                **{f"v2-extra-sft-s{s}": big for s in (1, 2, 3)}})  # fmt: skip
    vd.main(["h4", "--eval-dir", str(ev2), "--out", str(tmp_path / "c.json")])
    assert json.loads((tmp_path / "c.json").read_text())["H4"]["status"] == "not_supported"
    # one seed with no gain fails the every-seed clause even if the mean clears +2
    ev3 = tmp_path / "seed"
    _arms(ev3, {"v4-grpo-s1": int(0.07 * n), "v4-grpo-s2": int(0.07 * n), "v4-grpo-s3": 0,
                **{f"v2-extra-sft-s{s}": 0 for s in (1, 2, 3)}})  # fmt: skip
    vd.main(["h4", "--eval-dir", str(ev3), "--out", str(tmp_path / "s.json")])
    h4 = json.loads((tmp_path / "s.json").read_text())["H4"]
    assert h4["vs_parent"]["difference"] >= 0.02 and h4["status"] == "not_supported"
    # a missing run leaves H4 undecided, never passed
    ev4 = tmp_path / "missing"
    _arms(ev4, {"v4-grpo-s1": big, "v2-extra-sft-s1": 0})
    vd.main(["h4", "--eval-dir", str(ev4), "--out", str(tmp_path / "m.json")])
    assert json.loads((tmp_path / "m.json").read_text())["H4"]["status"] == "undecided"


def test_select_headroom_plateau(tmp_path):
    ev = tmp_path / "ev"
    base = _base(1000)
    for lr, k in (("1e-06", 5), ("3e-06", 30), ("1e-05", 30)):
        _write(ev, f"pilot-{lr}", "val_heldout_shape", _flip(base, k, 1))
    vd.main(
        [
            "select",
            "--eval-dir",
            str(ev),
            "--arm",
            "grpo",
            "--out",
            str(tmp_path / "s.json"),
            "--candidates",
            "1e-06:pilot-1e-06",
            "3e-06:pilot-3e-06",
            "1e-05:pilot-1e-05",
        ]
    )
    assert json.loads((tmp_path / "s.json").read_text())["selected"]["lr"] == 3e-6  # tie -> lower

    def samp(k, p1, pk, temp=1.0):
        return {"sampled": {"samples_per_prompt": k, "sampled_pass@1": {"value": p1},
                            f"pass@{k}": {"value": pk}},
                "decode": {"sampled": {"temperature": temp}}}  # fmt: skip

    _write(ev, "good", "val_heldout_shape", base, summary=samp(16, 0.70, 0.80))
    _write(ev, "flat", "val_heldout_shape", base, summary=samp(16, 0.70, 0.73))
    _write(ev, "wrongk", "val_heldout_shape", base, summary=samp(4, 0.70, 0.80))
    args = ["headroom", "--eval-dir", str(ev), "--out", str(tmp_path / "h.json"), "--name"]
    assert vd.main([*args, "good"]) == 0
    assert vd.main([*args, "flat"]) == 5 and vd.main([*args, "wrongk"]) == 5
    for n, k in ((375, 0), (750, 40), (1500, 60), (3000, 65)):
        _write(ev, f"sft-{n}", "val_heldout_shape", _flip(base, k, 2))
    vd.main(
        [
            "plateau",
            "--eval-dir",
            str(ev),
            "--out",
            str(tmp_path / "p.json"),
            "--points",
            "375000:sft-375",
            "750000:sft-750",
            "1500000:sft-1500",
            "3000000:sft-3000",
        ]
    )
    p = json.loads((tmp_path / "p.json").read_text())
    assert p["plateaued"] and abs(p["last_doubling_gain"] - 0.005) < 1e-9


def test_replication_rule(tmp_path):
    base = _base(3000)
    s = "test_extrapolation_v4x"

    def arms(ev, ctrl_drop, grpo_drop):
        _write(ev, "v2-sft-long", s, base)
        right = [i for i, c in enumerate(base) if c]
        for k in (1, 2, 3):
            g, x = list(base), list(base)
            for i in right[:ctrl_drop]:
                x[i] = False
            for i in right[:grpo_drop]:
                g[i] = False
            _write(ev, f"v4-grpo-s{k}", s, g)
            _write(ev, f"v2-extra-sft-s{k}", s, x)

    arms(tmp_path / "a", ctrl_drop=90, grpo_drop=5)
    vd.main(["replication", "--eval-dir", str(tmp_path / "a"), "--out", str(tmp_path / "a.json")])
    assert json.loads((tmp_path / "a.json").read_text())["status"] == "replicated"
    arms(tmp_path / "b", ctrl_drop=5, grpo_drop=5)
    vd.main(["replication", "--eval-dir", str(tmp_path / "b"), "--out", str(tmp_path / "b.json")])
    assert json.loads((tmp_path / "b.json").read_text())["status"] == "not_replicated"
    vd.main(
        ["replication", "--eval-dir", str(tmp_path / "none"), "--out", str(tmp_path / "c.json")]
    )
    assert json.loads((tmp_path / "c.json").read_text())["status"] == "undecided"
