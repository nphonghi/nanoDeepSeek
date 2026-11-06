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
    merge_and_unload_lora,
)
from data import get_batch, vocab_size, encode, decode


def main():
    # Detect device and print status
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Fine-tuning nanoDeepSeek with LoRA on device: {device}")

    # Instantiate base NanoDeepSeek model
    model_cfg = ModelConfig(vocab_size=vocab_size)
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
        t0 = time.time()
        x, y = get_batch("train", batch_size=32, block_size=64, device=str(device))
        outputs = model(x, labels=y)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar_len = 20
        progress = step / 100
        filled = int(bar_len * progress)
        bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - (1 if filled < bar_len else 0))
        sys.stdout.write(f"\r[{bar}] {progress * 100:5.1f}% | LoRA Step {step:3d}/100 | Loss: {loss.item():.4f}")
        sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"LoRA fine-tuning completed in {time.time() - start_time:.2f} seconds")

    # Merge adapter parameters into base weights and unwrap LoRA layers
    model = merge_and_unload_lora(model)
    print("Merged LoRA weights into base model successfully")

    # Save fine-tuned checkpoint for downstream inference
    lora_checkpoint = Path(__file__).resolve().parent / "lora_model.pt"
    torch.save(model.state_dict(), lora_checkpoint)
    print(f"Saved fine-tuned LoRA model checkpoint to {lora_checkpoint.name}")


if __name__ == "__main__":
    main()
