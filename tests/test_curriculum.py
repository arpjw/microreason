"""Curriculum phases in the packed stream and the stage config."""

import json

import pytest
from pydantic import ValidationError
from test_training import make_dataset, random_docs

from microreason.config import StageConfig
from microreason.train.data import PackedStream, Source

BOS, EOS = 1, 2


@pytest.fixture
def sources(tmp_path):
    return [
        Source("easy", make_dataset(tmp_path / "e", random_docs(1, 400, 5, 20), name="easy"), 1),
        Source("hard", make_dataset(tmp_path / "h", random_docs(2, 400, 20, 60), name="hard"), 1),
        Source("text", make_dataset(tmp_path / "t", random_docs(3, 400, 30, 90), name="text"), 1),
    ]


SCHEDULE = [(20_000, [0.6, 0.25, 0.15]), (50_000, [0.1, 0.75, 0.15])]


def _drain(stream, tokens):
    while sum(stream.tokens) < tokens:
        stream.next_sequence()


def test_phases_follow_their_own_weights_and_are_logged(sources):
    s = PackedStream(sources, 64, 0, BOS, EOS, SCHEDULE)
    _drain(s, 19_000)
    share = [t / sum(s.phase_tokens) for t in s.phase_tokens]
    assert share == pytest.approx([0.6, 0.25, 0.15], abs=0.02)
    assert s.phase == 0 and s.phase_log == []
    _drain(s, 49_000)
    assert s.phase == 1
    (entry,) = s.phase_log
    assert entry["phase"] == 1 and 20_000 <= entry["at_tokens"] < 20_200
    share = [t / sum(s.phase_tokens) for t in s.phase_tokens]
    assert share == pytest.approx([0.1, 0.75, 0.15], abs=0.02)
    assert s.mixture_report()["curriculum"]["current_phase"] == 1


def test_last_phase_continues_past_its_bound(sources):
    s = PackedStream(sources, 64, 0, BOS, EOS, SCHEDULE)
    _drain(s, 60_000)
    assert s.phase == 1


def test_curriculum_state_roundtrip_mid_phase(sources):
    s = PackedStream(sources, 64, 0, BOS, EOS, SCHEDULE)
    _drain(s, 25_000)
    state = json.loads(json.dumps(s.state_dict()))
    want = [s.next_sequence() for _ in range(300)]
    r = PackedStream(sources, 64, 0, BOS, EOS, SCHEDULE)
    r.load_state_dict(state)
    assert [r.next_sequence() for _ in range(300)] == want
    assert r.phase_log == s.phase_log
    other = PackedStream(sources, 64, 0, BOS, EOS, [(50_000, [1, 1, 1])])
    with pytest.raises(ValueError):
        other.load_state_dict(state)


@pytest.mark.parametrize(
    "schedule",
    [[(100, [1, 1, 1]), (50, [1, 1, 1])], [(100, [1, 1])], [(100, [0, 0, 0])],
     [(100, [-1, 1, 1])]],
)  # fmt: skip
def test_bad_schedules_rejected(sources, schedule):
    with pytest.raises(ValueError):
        PackedStream(sources, 64, 0, BOS, EOS, schedule)


def _stage(curriculum):
    return StageConfig.model_validate(
        dict(
            stage="smoke", model="m.yaml", sequence_length=32, effective_batch_tokens=256,
            max_steps=10, optimizer={"peak_lr": 1e-3},
            mixture=[{"dataset": "a", "weight": 1}, {"dataset": "b", "weight": 1}],
            curriculum=curriculum,
        )
    )  # fmt: skip


def test_config_curriculum_validation():
    ok = _stage([{"until_fraction": 0.5, "weights": {"a": 1, "b": 0}},
                 {"until_fraction": 1.0, "weights": {"a": 0, "b": 1}}])  # fmt: skip
    assert len(ok.curriculum) == 2
    with pytest.raises(ValidationError, match="end at 1.0"):
        _stage([{"until_fraction": 0.5, "weights": {"a": 1, "b": 1}}])
    with pytest.raises(ValidationError, match="increase"):
        _stage([{"until_fraction": 0.6, "weights": {"a": 1, "b": 1}},
                {"until_fraction": 0.5, "weights": {"a": 1, "b": 1}}])  # fmt: skip
    with pytest.raises(ValidationError, match="exactly the mixture"):
        _stage([{"until_fraction": 1.0, "weights": {"a": 1}}])
