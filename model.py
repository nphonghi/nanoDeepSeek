import math
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig
from modules import (
    RMSNorm,
    MultiHeadLatentAttention,
    DeepSeekMoE,
    SwiGLU,
    MTPHead,
)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe_layer(layer_idx)

        # Pre-attention normalization and Multi-Head Latent Attention
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        # Pre-FFN normalization and either MoE or standard dense SwiGLU
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.is_moe:
            self.ffn = DeepSeekMoE(config.hidden_size, config.moe)
        else:
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Attention sub-layer with residual connection
        attn_out, present_kv = self.attn(self.attn_norm(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out

        # Feed-forward sub-layer (MoE or dense SwiGLU) with residual connection
        if self.is_moe:
            ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
            aux_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, present_kv, aux_loss


class NanoDeepSeek(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if config.mtp.enabled:
            self.mtp_head = MTPHead(
                config=config,
                num_predict_tokens=config.mtp.num_predict_tokens,
            )
        else:
            self.mtp_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        # Look up token embeddings from input token IDs
        x = self.embed_tokens(input_ids)

        present_kvs = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        # Pass representations through all transformer blocks
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv, aux_loss = layer(x, past_key_value=past_kv, use_cache=use_cache)
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                present_kvs.append(present_kv)

        # Apply final RMSNorm and compute language model logits
        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)

        loss = None
        mtp_logits = None

        # Compute next-token prediction loss and multi-token prediction loss
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = lm_loss + total_aux_loss

            # Add DeepSeek-V3 multi-token prediction loss
            if self.mtp_head is not None:
                mtp_logits = self.mtp_head(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    embed_tokens=self.embed_tokens,
                    lm_head=self.lm_head,
                )
                mtp_loss = self.mtp_head.compute_loss(mtp_logits, labels)
                loss = loss + self.config.mtp.mtp_loss_weight * mtp_loss

        elif self.mtp_head is not None:
            mtp_logits = self.mtp_head(
                hidden_states=hidden_states,
                input_ids=input_ids,
                embed_tokens=self.embed_tokens,
                lm_head=self.lm_head,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "mtp_logits": mtp_logits,
            "aux_loss": total_aux_loss,
            "past_key_values": present_kvs,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        # Switch to evaluation mode and initialize state
        self.eval()
        past_key_values = None
        generated = input_ids

        # Autoregressive token generation loop
        for _ in range(max_new_tokens):
            curr_input = generated[:, -1:] if past_key_values is not None else generated
            outputs = self.forward(curr_input, past_key_values=past_key_values, use_cache=True)
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply temperature, top-k filtering, and nucleus (top-p) sampling
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < val[:, [-1]]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                    logits[mask] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append newly generated token to ongoing sequence
            generated = torch.cat((generated, next_token), dim=1)

        return generated
