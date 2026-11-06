import math
from typing import List, Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze original base weight matrix
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Initialize low-rank adapter matrices matching device and dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize matrix A with Kaiming uniform and matrix B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute original base forward output
        base_out = self.base_layer(x)
        # Compute low-rank adapter delta: (dropout(x) @ A^T) @ B^T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        # Merge adapter weights into base layer for zero-overhead inference
        if not self.base_layer.weight.requires_grad:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w.to(self.base_layer.weight.device)


def apply_lora(
    model: nn.Module,
    target_modules: List[str] = ["q_down_proj", "kv_down_proj", "o_proj"],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> nn.Module:
    # Wrap target linear projection layers with low-rank adapters
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(target in child_name for target in target_modules):
                lora_layer = LoRALinear(
                    child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(module, child_name, lora_layer)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    # Freeze base model weights and only enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def merge_and_unload_lora(model: nn.Module) -> nn.Module:
    # Merge adapter weights into base linear layers and unwrap LoRALinear modules
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                child.merge_weights()
                setattr(module, child_name, child.base_layer)
    return model


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({ratio:.2f}%)")
