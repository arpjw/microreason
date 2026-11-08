"""MicroReason decoder-only Transformer (spec §3).

Pre-norm blocks with RMSNorm, rotary position embeddings on queries and keys, SwiGLU MLPs,
PyTorch scaled-dot-product attention, no biases, and input/output embeddings tied by reusing
the embedding matrix as the output projection. The same class builds the full and smoke models.

It is wrapped as a Hugging Face PreTrainedModel so save_pretrained / from_pretrained work.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from microreason.config import ModelConfig
from microreason.prompts import IGNORE_INDEX


class MicroReasonConfig(PretrainedConfig):
    model_type = "microreason"

    def __init__(
        self,
        name: str = "microreason",
        n_layers: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        head_dim: int = 64,
        mlp_hidden: int = 1408,
        vocab_size: int = 8192,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        init_std: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.name = name
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.mlp_hidden = mlp_hidden
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.init_std = init_std
        kwargs.setdefault("tie_word_embeddings", False)  # tying is done by hand, see forward
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_model_config(cls, m: ModelConfig, **kwargs) -> MicroReasonConfig:
        if not m.tie_embeddings or m.bias or m.dropout != 0.0:
            raise ValueError("v1 model requires tied embeddings, no bias and no dropout")
        return cls(
            name=m.name,
            n_layers=m.n_layers,
            d_model=m.d_model,
            n_heads=m.n_heads,
            n_kv_heads=m.n_kv_heads,
            head_dim=m.head_dim,
            mlp_hidden=m.mlp_hidden,
            vocab_size=m.vocab_size,
            max_seq_len=m.max_seq_len,
            rope_theta=m.rope_theta,
            **kwargs,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, D): pairs dimension i with i + D/2
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (B, 1, T, D) gathered at each token's position."""
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Per-layer key/value buffers for incremental decoding, preallocated to `max_len` keys.

    Buffers are created on first write with the dtype the keys arrive in (bf16 under autocast).
    `length` counts filled key positions; the model advances it after every layer has written.
    """

    def __init__(self, n_layers: int, max_len: int):
        self.max_len = max_len
        self.length = 0
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        b, h, t, d = k.shape
        end = self.length + t
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        if self.k[layer] is None:
            self.k[layer] = k.new_zeros((b, h, self.max_len, d))
            self.v[layer] = v.new_zeros((b, h, self.max_len, d))
        self.k[layer][:, :, self.length : end] = k
        self.v[layer][:, :, self.length : end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: MicroReasonConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask, cache: KVCache | None = None, layer: int = 0):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.write(layer, k, v)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(y.transpose(1, 2).reshape(b, t, self.n_heads * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, cfg: MicroReasonConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: MicroReasonConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask, cache=None, layer=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, cache, layer)
        return x + self.mlp(self.mlp_norm(x))


class MicroReasonForCausalLM(PreTrainedModel):
    config_class = MicroReasonConfig
    base_model_prefix = "microreason"
    _no_split_modules = ["Block"]

    def __init__(self, config: MicroReasonConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        # RoPE tables are derived from the config, not stored: Transformers 5 materializes
        # models from a meta device, which would leave non-persistent buffers uninitialized.
        self._rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.post_init()

    def rope_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._rope:
            c = self.config
            self._rope[device] = rope_cache(c.max_seq_len, c.head_dim, c.rope_theta, device)
        return self._rope[device]

    # Hugging Face hooks -------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # Residual output projections get a depth-scaled init (GPT-2 style).
            scale = getattr(module, "_residual_scale", 1.0)
            nn.init.normal_(module.weight, mean=0.0, std=std * scale)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def post_init(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            block.attn.o._residual_scale = scale
            block.mlp.down._residual_scale = scale
        super().post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value

    # Forward ------------------------------------------------------------------------------

    @staticmethod
    def build_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, T) key mask of 1/0 -> (B, 1, T, T) boolean causal mask.

        Every query may always attend to itself, so fully padded rows stay finite; their
        outputs are never used because their labels are ignored.
        """
        b, t = attention_mask.shape
        causal = torch.ones(t, t, dtype=torch.bool, device=attention_mask.device).tril()
        keys = attention_mask.bool()[:, None, None, :]
        eye = torch.eye(t, dtype=torch.bool, device=attention_mask.device)
        return (causal & keys) | eye

    @staticmethod
    def build_cached_attention_mask(attention_mask: torch.Tensor, start: int, t: int):
        """Mask for `t` new queries at key positions [start, start + t) over all keys so far.

        attention_mask is the (B, start + t) key mask. Rows equal the last `t` rows of
        build_attention_mask(attention_mask), so cached and uncached decoding see the same keys.
        """
        dev = attention_mask.device
        q_idx = torch.arange(start, start + t, device=dev)[:, None]
        k_idx = torch.arange(attention_mask.shape[1], device=dev)[None, :]
        keys = attention_mask.bool()[:, None, None, :]
        return ((k_idx <= q_idx) & keys) | (k_idx == q_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
        **_: object,
    ) -> CausalLMOutput:
        """With `cache`, input_ids holds only the new tokens and attention_mask covers every key
        so far (cached plus new); positions come from the full mask, as without a cache."""
        b, t = input_ids.shape
        start = 0 if cache is None else cache.length
        if start + t > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {start + t} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if cache is not None and attention_mask is None:
            raise ValueError("cached decoding needs the full attention mask")
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(t, device=input_ids.device).expand(b, t)
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -t:]
        cos_t, sin_t = self.rope_tables(input_ids.device)
        cos = cos_t[position_ids].unsqueeze(1)
        sin = sin_t[position_ids].unsqueeze(1)
        if attention_mask is None:
            mask = None
        elif cache is None:
            mask = self.build_attention_mask(attention_mask)
        else:
            mask = self.build_cached_attention_mask(attention_mask, start, t)

        x = self.embed(input_ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask, cache, i)
        if cache is not None:
            cache.length = start + t
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight)  # tied output projection

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return CausalLMOutput(loss=loss, logits=logits)


def build_model(m: ModelConfig, **overrides) -> MicroReasonForCausalLM:
    return MicroReasonForCausalLM(MicroReasonConfig.from_model_config(m, **overrides))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
