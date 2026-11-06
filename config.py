from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MoEConfig:
    enabled: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 1
    num_shared_experts: int = 1
    expert_hidden_size: int = 256
    aux_loss_weight: float = 0.01
    aux_loss_free: bool = False  # Set to True for DeepSeek-V3 auxiliary-loss-free load balancing
    bias_update_rate: float = 0.001


@dataclass
class MTPConfig:
    enabled: bool = True
    num_predict_tokens: int = 1
    mtp_loss_weight: float = 0.3


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_down_proj", "kv_down_proj", "o_proj"]
    )


@dataclass
class ModelConfig:
    vocab_size: int = 65
    hidden_size: int = 128
    num_hidden_layers: int = 4
    max_position_embeddings: int = 256
    
    num_attention_heads: int = 4
    head_dim: int = 32
    kv_lora_rank: int = 32
    q_lora_rank: int = 48
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    rope_theta: float = 10000.0
    
    intermediate_size: int = 256
    moe_layer_freq: int = 1
    first_moe_layer: int = 1
    moe: MoEConfig = field(default_factory=MoEConfig)
    
    mtp: MTPConfig = field(default_factory=MTPConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.moe.enabled or layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0


@dataclass
class TrainConfig:
    batch_size: int = 32
    block_size: int = 64
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_steps: int = 500
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_steps: int = 20
    device: str = "auto"
