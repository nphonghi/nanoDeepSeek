from typing import Tuple, Optional
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Calculate inverse frequencies for rotary transformation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cosine and sine frequency cache
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dynamically expand cache if sequence length exceeds cached limit
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        # Broadcast cosine and sine to shape (1, 1, seq_len, dim)
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # Apply standard 2D rotation formula to complex-paired elements
    return (x * cos) + (rotate_half(x) * sin)
