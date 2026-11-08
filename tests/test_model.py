from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from microreason.config import load_model_config
from microreason.model import (
    MicroReasonForCausalLM,
    apply_rope,
    build_model,
    count_parameters,
    rope_cache,
)
from microreason.prompts import IGNORE_INDEX

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ATOL = 1e-5


@pytest.fixture(scope="module")
def smoke():
    torch.manual_seed(0)
    return build_model(load_model_config(CONFIGS / "model_smoke.yaml")).eval()


def test_full_model_count_matches_analytic():
    mc = load_model_config(CONFIGS / "model_43m.yaml")
    m = build_model(mc)
    assert count_parameters(m) == mc.parameter_count() == 42_742_272


def test_smoke_model_count_matches_analytic(smoke):
    mc = load_model_config(CONFIGS / "model_smoke.yaml")
    assert count_parameters(smoke) == mc.parameter_count() == 3_344_064


def test_no_biases_and_tied_output(smoke):
    names = [n for n, _ in smoke.named_parameters()]
    assert not any(n.endswith("bias") for n in names)
    assert not any("lm_head" in n for n in names)


def test_initial_loss_near_uniform(smoke):
    x = torch.randint(9, 8192, (4, 64))
    loss = smoke(x, labels=x).loss.item()
    assert abs(loss - torch.log(torch.tensor(8192.0)).item()) < 0.5


def test_causality(smoke):
    torch.manual_seed(1)
    x = torch.randint(9, 8192, (2, 48))
    y = x.clone()
    t = 20
    y[:, t:] = torch.randint(9, 8192, (2, 48 - t))
    with torch.no_grad():
        a, b = smoke(x).logits, smoke(y).logits
    assert torch.allclose(a[:, :t], b[:, :t], atol=ATOL)
    assert not torch.allclose(a[:, t:], b[:, t:], atol=ATOL)


def test_right_padding_matches_unpadded(smoke):
    x = torch.randint(9, 8192, (1, 30))
    padded = torch.cat([x, torch.zeros(1, 10, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 30), torch.zeros(1, 10)], dim=1).long()
    with torch.no_grad():
        a = smoke(x).logits
        b = smoke(padded, attention_mask=mask).logits[:, :30]
    assert torch.allclose(a, b, atol=ATOL)


def test_left_padding_positions_and_mask(smoke):
    x = torch.randint(9, 8192, (1, 30))
    padded = torch.cat([torch.zeros(1, 7, dtype=torch.long), x], dim=1)
    mask = torch.cat([torch.zeros(1, 7), torch.ones(1, 30)], dim=1).long()
    with torch.no_grad():
        a = smoke(x).logits
        b = smoke(padded, attention_mask=mask).logits[:, 7:]
        assert torch.isfinite(smoke(padded, attention_mask=mask).logits).all()
    assert torch.allclose(a, b, atol=ATOL)


def test_batch_rows_independent(smoke):
    x = torch.randint(9, 8192, (3, 25))
    with torch.no_grad():
        batched = smoke(x).logits
        single = smoke(x[1:2]).logits
    assert torch.allclose(batched[1:2], single, atol=ATOL)


def test_loss_masking(smoke):
    torch.manual_seed(2)
    x = torch.randint(9, 8192, (2, 32))
    labels = x.clone()
    labels[:, :12] = IGNORE_INDEX
    out = smoke(x, labels=labels)
    logits = out.logits[:, :-1].reshape(-1, 8192)
    manual = F.cross_entropy(logits, labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
    assert torch.allclose(out.loss, manual)


def test_ignored_positions_get_zero_gradient():
    torch.manual_seed(3)
    model = build_model(load_model_config(CONFIGS / "model_smoke.yaml"))
    x = torch.randint(9, 8192, (2, 32))
    labels = x.clone()
    labels[:, :12] = IGNORE_INDEX  # prompt
    labels[:, 28:] = IGNORE_INDEX  # padding
    out = model(x, labels=labels)
    out.logits.retain_grad()
    out.loss.backward()
    g = out.logits.grad.abs().sum(-1)  # (B, T); position i predicts labels[i + 1]
    assert torch.all(g[:, :11] == 0)
    assert torch.all(g[:, 27:] == 0)
    assert torch.all(g[:, 11:27] > 0)


def _reference_rope(x, pos, theta=10000.0):
    """Loop implementation of the half-split rotary convention."""
    d = x.shape[-1]
    out = torch.empty_like(x)
    for i in range(d // 2):
        freq = 1.0 / theta ** (2 * i / d)
        ang = pos * freq
        c, s = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
        a, b = x[..., i], x[..., i + d // 2]
        out[..., i] = a * c - b * s
        out[..., i + d // 2] = a * s + b * c
    return out


def test_rope_matches_reference():
    cos, sin = rope_cache(32, 64, 10000.0)
    x = torch.randn(64)
    for pos in (0, 1, 7, 31):
        got = apply_rope(
            x[None, None, None], cos[pos][None, None, None], sin[pos][None, None, None]
        )
        assert torch.allclose(got.flatten(), _reference_rope(x, pos), atol=1e-5)


def test_rope_depends_only_on_relative_position():
    cos, sin = rope_cache(64, 64, 10000.0)
    q, k = torch.randn(64), torch.randn(64)

    def rot(v, p):
        return apply_rope(v[None, None, None], cos[p][None, None, None], sin[p][None, None, None])

    a = (rot(q, 10) * rot(k, 3)).sum()
    b = (rot(q, 40) * rot(k, 33)).sum()
    assert torch.allclose(a, b, atol=1e-4)


def test_save_load_roundtrip(smoke, tmp_path):
    smoke.save_pretrained(tmp_path)
    loaded = MicroReasonForCausalLM.from_pretrained(tmp_path).eval()
    x = torch.randint(9, 8192, (2, 20))
    with torch.no_grad():
        assert torch.equal(smoke(x).logits, loaded(x).logits)
    assert count_parameters(loaded) == count_parameters(smoke)


def test_rejects_sequence_longer_than_context(smoke):
    with pytest.raises(ValueError, match="max_seq_len"):
        smoke(torch.zeros(1, 513, dtype=torch.long))


def test_backward_finite_cpu():
    model = build_model(load_model_config(CONFIGS / "model_smoke.yaml"))
    x = torch.randint(9, 8192, (2, 64))
    model(x, labels=x).loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_matches_cpu(smoke):
    x = torch.randint(9, 8192, (2, 32))
    model = build_model(load_model_config(CONFIGS / "model_smoke.yaml")).eval()
    with torch.no_grad():
        cpu = model(x).logits
        mps = model.to("mps")(x.to("mps")).logits.cpu()
    assert torch.allclose(cpu, mps, atol=1e-3)
    loss = model(x.to("mps"), labels=x.to("mps")).loss
    loss.backward()
    assert torch.isfinite(loss).item()
