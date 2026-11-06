import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MoEConfig


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish-Gated Linear Unit activation
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_x = x.view(-1, x.size(-1))
        logits = self.gate(flat_x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        # Select top-k experts per token and re-normalize routing weights
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        # Compute auxiliary loss to encourage balanced routing across all experts
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        fraction_routed = expert_mask.float().mean(dim=0)
        mean_probability = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(fraction_routed * mean_probability)

        return topk_indices, topk_weights, aux_loss


class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_size: int, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        self.aux_loss_weight = config.aux_loss_weight

        # Shared experts are always active for all tokens to retain common knowledge
        if self.num_shared_experts > 0:
            shared_dim = config.expert_hidden_size * self.num_shared_experts
            self.shared_experts = SwiGLU(hidden_size, shared_dim)
        else:
            self.shared_experts = None

        # Routed experts are selectively activated based on token features
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size, config.expert_hidden_size)
            for _ in range(self.num_experts)
        ])

        self.router = MoEGate(hidden_size, self.num_experts, self.top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.view(-1, D)

        shared_out = self.shared_experts(flat_x) if self.shared_experts is not None else 0.0

        topk_indices, topk_weights, aux_loss = self.router(flat_x)

        # Dispatch tokens to selected experts and accumulate weighted outputs
        routed_out = torch.zeros_like(flat_x)
        for expert_id in range(self.num_experts):
            token_mask = (topk_indices == expert_id)
            if not token_mask.any():
                continue
            
            token_idx, k_idx = token_mask.nonzero(as_tuple=True)
            expert_tokens = flat_x[token_idx]
            weights = topk_weights[token_idx, k_idx].unsqueeze(-1)

            expert_out = self.experts[expert_id](expert_tokens) * weights
            routed_out.index_add_(0, token_idx, expert_out)

        total_out = (shared_out + routed_out).view(B, L, D)
        scaled_aux_loss = self.aux_loss_weight * aux_loss
        return total_out, scaled_aux_loss
