from contextlib import nullcontext
import math
import time
import sys
from pathlib import Path
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig, TrainConfig
from model import NanoDeepSeek
from data import get_batch, vocab_size, encode, decode


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr: float, base_lr: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"


@torch.no_grad()
def estimate_loss(model: NanoDeepSeek, train_cfg: TrainConfig, device: torch.device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_cfg.eval_steps)
        for k in range(train_cfg.eval_steps):
            x, y = get_batch(split, train_cfg.batch_size, train_cfg.block_size, device=str(device))
            outputs = model(x, labels=y)
            losses[k] = outputs["loss"].item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    # Device selection and hardware detection
    device = get_device()
    device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
    print(f"Training nanoDeepSeek on device: {device} ({device_type})")

    # Mixed precision context setup (autocast)
    ptdtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda" and ptdtype != torch.float32) else nullcontext()

    # Training and model configurations
    train_cfg = TrainConfig(
        batch_size=16,
        block_size=128,
        max_steps=200,
        warmup_steps=20,
        learning_rate=5e-4,
        eval_interval=50,
        eval_steps=10,
    )

    model_cfg = ModelConfig(vocab_size=vocab_size)

    # Instantiate NanoDeepSeek model
    model = NanoDeepSeek(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Optimizer configuration with parameter weight decay separation
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("weight" in n and "norm" not in n and "embed" not in n)]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n or "embed" in n)]
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate)

    scheduler = get_cosine_scheduler(
        optimizer,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=train_cfg.max_steps,
        min_lr=train_cfg.min_learning_rate,
        base_lr=train_cfg.learning_rate,
    )

    # Main pretraining iteration loop
    model.train()
    start_time = time.time()
    print("Starting training loop...", flush=True)

    for step in range(1, train_cfg.max_steps + 1):
        t0 = time.time()
        # Fetch training data batch
        x, y = get_batch("train", train_cfg.batch_size, train_cfg.block_size, device=str(device))

        # Forward pass under mixed precision context
        with ctx:
            outputs = model(x, labels=y)
            loss = outputs["loss"]

        # Backward pass and optimizer step with gradient clipping
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Track throughput metrics and progress display
        dt = time.time() - t0
        tokens_per_sec = (train_cfg.batch_size * train_cfg.block_size) / max(1e-6, dt)
        remaining_steps = train_cfg.max_steps - step
        eta = remaining_steps * dt

        # Visual progress bar rendering
        bar_len = 20
        progress = step / train_cfg.max_steps
        filled = int(bar_len * progress)
        bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - (1 if filled < bar_len else 0))
        lr = scheduler.get_last_lr()[0]
        aux_loss = outputs["aux_loss"].item()

        progress_str = (
            f"\r[{bar}] {progress * 100:5.1f}% | "
            f"Step {step:3d}/{train_cfg.max_steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Aux: {aux_loss:.4f} | "
            f"{tokens_per_sec:4.0f} tok/s | "
            f"ETA: {format_time(eta)}"
        )
        sys.stdout.write(progress_str)
        sys.stdout.flush()

        # Validation evaluation and text generation check
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            sys.stdout.write("\n")
            print(f"[Step {step}] Evaluating on validation set...", flush=True)
            losses = estimate_loss(model, train_cfg, device)
            print(f"Validation loss: train {losses['train']:.4f}, val {losses['val']:.4f}", flush=True)

            model.eval()
            prompt = "\n"
            prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8)
            print(f"Generated text sample:\n{decode(gen_ids[0].tolist())}\n", flush=True)
            model.train()

    sys.stdout.write("\n")
    print(f"Training completed in {time.time() - start_time:.1f} seconds", flush=True)

    # Save trained base model checkpoint for downstream fine-tuning and inference
    checkpoint_path = Path(__file__).resolve().parent / "base_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved base model checkpoint to {checkpoint_path.name}", flush=True)


if __name__ == "__main__":
    train()
