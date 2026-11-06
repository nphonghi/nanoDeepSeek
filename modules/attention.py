import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        
        # Total dimension of each query and key head
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        
        # Low-rank query compression and up-projections
        self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_attention_heads * self.qk_head_dim, bias=False)
        
        # Low-rank key-value joint compression
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        
        # Unpack key and value from compressed latent vector
        self.k_up_proj = nn.Linear(kv_lora_rank, num_attention_heads * qk_nope_head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_attention_heads * v_head_dim, bias=False)
        
        # Decoupled rotary key projection directly from input
        self.k_rope_proj = nn.Linear(hidden_size, qk_rope_head_dim, bias=False)
        
        # Final output projection
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
        
        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(qk_rope_head_dim, max_seq_len=max_seq_len, base=rope_theta)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        # Compress and project query heads
        q_compressed = self.q_norm(self.q_down_proj(x))
        q = self.q_up_proj(q_compressed)
        q = q.view(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)
        
        # Split query into non-positional and positional sub-vectors
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Joint low-rank compression of keys and values into latent vector c_t^{KV}
        # In DeepSeek MLA, this compressed latent representation is the ONLY thing cached (along with decoupled RoPE key)
        kv_compressed = self.kv_norm(self.kv_down_proj(x))  # [B, L, kv_lora_rank]
        
        # Decoupled rotary key projection directly from input x (shared across all heads)
        k_rope = self.k_rope_proj(x).unsqueeze(1)  # [B, 1, L, qk_rope_head_dim]
        
        # Rotary position embeddings
        # Determine sequence positions based on cached past sequence length
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        total_len = past_len + L
        
        cos, sin = self.rotary_emb(q_rope, total_len)
        cos = cos[:, :, past_len:total_len, :]
        sin = sin[:, :, past_len:total_len, :]
        
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)
        
        # MLA KV CACHE: Cache ONLY the compressed latent vector and decoupled RoPE key!
        # Memory footprint per token: (kv_lora_rank + qk_rope_head_dim) instead of 2 * num_heads * head_dim.
        # This achieves massive memory compression (e.g. 56x in DeepSeek-V3).
        if past_key_value is not None:
            past_kv, past_k_rope = past_key_value
            cached_kv = torch.cat([past_kv, kv_compressed], dim=1)
            cached_k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        else:
            cached_kv = kv_compressed
            cached_k_rope = k_rope
            
        present_key_value = (cached_kv, cached_k_rope) if use_cache else None
        
        # Decompress keys and values from the latent cache for attention calculation
        k_nope = self.k_up_proj(cached_kv).view(B, total_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.v_up_proj(cached_kv).view(B, total_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        k_rope_expanded = cached_k_rope.expand(B, self.num_heads, total_len, self.qk_rope_head_dim)
        
        # Assemble full queries and keys: [B, num_heads, seq_len, qk_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)
        
        # Scaled Dot-Product Attention
        # PyTorch MPS backend has an internal Metal kernel issue when qk_head_dim != v_head_dim.
        # Fall back to exact dot-product attention when running on MPS or when head dimensions differ.
        dropout_p = self.dropout.p if self.training else 0.0
        if x.device.type == "mps" or q.size(-1) != v.size(-1):
            scale = 1.0 / math.sqrt(q.size(-1))
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if L > 1 and past_len > 0:
                q_idx = torch.arange(L, device=x.device).unsqueeze(1)
                k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
                causal_mask = (k_idx <= (past_len + q_idx)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal_mask, float("-inf"))
            elif (L > 1) and (past_len == 0):
                mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                probs = F.dropout(probs, p=dropout_p)
            attn_out = torch.matmul(probs, v)
        else:
            if L > 1 and past_len > 0:
                # Chunked decoding with past cache: explicit causal mask
                q_idx = torch.arange(L, device=x.device).unsqueeze(1)
                k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
                causal_mask = (k_idx <= (past_len + q_idx)).unsqueeze(0).unsqueeze(0)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=causal_mask,
                    dropout_p=dropout_p,
                )
            else:
                # Prefill or single-token decoding
                is_causal = (L > 1) and (past_len == 0)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=is_causal,
                    dropout_p=dropout_p,
                )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.v_head_dim)
        out = self.o_proj(attn_out)
        return out, present_key_value
