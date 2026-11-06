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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        # Compress and project query heads
        q_compressed = self.q_norm(self.q_down_proj(x))
        q = self.q_up_proj(q_compressed)
        q = q.view(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)
        
        # Split query into non-positional and positional sub-vectors
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Compress and project keys and values
        kv_compressed = self.kv_norm(self.kv_down_proj(x))
        k_nope = self.k_up_proj(kv_compressed).view(B, L, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.v_up_proj(kv_compressed).view(B, L, self.num_heads, self.v_head_dim).transpose(1, 2)
        
        # Project decoupled positional key
        k_rope = self.k_rope_proj(x).unsqueeze(1)
        
        # Apply rotary position embeddings
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        total_len = past_len + L
        
        cos, sin = self.rotary_emb(q_rope, total_len)
        cos = cos[:, :, past_len:total_len, :]
        sin = sin[:, :, past_len:total_len, :]
        
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)
        k_rope = k_rope.expand(B, self.num_heads, L, self.qk_rope_head_dim)
        
        # Reconstruct full query and key representations
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # Update key-value cache during generation
        if past_key_value is not None:
            past_k, past_v, _ = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_key_value = (k, v, kv_compressed) if use_cache else None
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal masking
        if L > 1:
            causal_mask = torch.triu(torch.full((L, total_len), float('-inf'), device=x.device), diagonal=past_len + 1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.v_head_dim)
        
        out = self.o_proj(attn_out)
        return out, present_key_value
