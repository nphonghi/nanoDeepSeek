from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class MTPHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predict_tokens: int = 1,
        embedding_weight: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.num_predict_tokens = num_predict_tokens
        
        # Projection layers mapping current representations to future token depths
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_predict_tokens)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(hidden_size)
            for _ in range(num_predict_tokens)
        ])
        
        # Output heads predicting future vocabulary tokens
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_predict_tokens)
        ])
        if embedding_weight is not None:
            for head in self.output_heads:
                head.weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        predictions = []
        for i in range(self.num_predict_tokens):
            h = self.projections[i](hidden_states)
            h = self.norms[i](h)
            logits = self.output_heads[i](h)
            predictions.append(logits)
        return predictions

    def compute_loss(
        self,
        mtp_logits_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = 0.0
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Compute prediction loss for tokens at positions t+2, t+3 and beyond
        for depth, logits in enumerate(mtp_logits_list, start=1):
            shift_logits = logits[:, :-(depth + 1), :].contiguous()
            shift_labels = labels[:, (depth + 1):].contiguous()

            if shift_logits.size(1) > 0:
                loss = loss_fct(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss = total_loss + loss

        return total_loss / max(1, len(mtp_logits_list))
