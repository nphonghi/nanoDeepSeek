import time
import sys
from pathlib import Path
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig
from model import NanoDeepSeek
from modules.lora import (
    apply_lora,
    mark_only_lora_as_trainable,
    print_trainable_parameters,
)
from data import get_batch, vocab_size, encode, decode


def main():
    # Detect device and print status
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Fine-tuning nanoDeepSeek with LoRA on device: {device}")

    # Instantiate base NanoDeepSeek model
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        max_position_embeddings=256,
        num_attention_heads=4,
        head_dim=32,
        kv_lora_rank=32,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
    )
    model = NanoDeepSeek(model_cfg).to(device)

    # Load pretrained base model checkpoint if available
    base_checkpoint = Path(__file__).resolve().parent / "base_model.pt"
    if base_checkpoint.exists():
        model.load_state_dict(torch.load(base_checkpoint, map_location=device))
        print(f"Loaded pretrained base model from {base_checkpoint.name}")
    else:
        print("No base model checkpoint found, starting from random initialization")

    print("Base model parameter count:")
    print_trainable_parameters(model)

    # Attach low-rank adapters to MLA projection matrices
    target_projections = ["q_down_proj", "kv_down_proj", "o_proj"]
    model = apply_lora(
        model,
        target_modules=target_projections,
        r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
    )

    # Freeze base weights and train only LoRA parameters
    mark_only_lora_as_trainable(model)
    print("After applying LoRA and freezing base weights:")
    print_trainable_parameters(model)

    # Optimize only trainable adapter parameters
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    start_time = time.time()

    for step in range(1, 101):
        x, y = get_batch("train", batch_size=32, block_size=64, device=str(device))
        outputs = model(x, labels=y)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"LoRA Step {step:3d}/100 | Loss: {loss.item():.4f}")

    print(f"LoRA fine-tuning completed in {time.time() - start_time:.2f} seconds")

    # Merge adapter parameters into base weights
    for module in model.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights()
    print("Merged LoRA weights into base model successfully")

    # Save fine-tuned checkpoint for downstream inference
    lora_checkpoint = Path(__file__).resolve().parent / "lora_model.pt"
    torch.save(model.state_dict(), lora_checkpoint)
    print(f"Saved fine-tuned LoRA model checkpoint to {lora_checkpoint.name}")


if __name__ == "__main__":
    main()
