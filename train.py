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
    device = get_device()
    print(f"Training nanoDeepSeek on device: {device}")

    train_cfg = TrainConfig(
        batch_size=32,
        block_size=64,
        max_steps=150,
        warmup_steps=20,
        learning_rate=1e-3,
        eval_interval=50,
        eval_steps=10,
    )

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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Exclude 1D tensors (biases, norms, embeddings) from weight decay
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

    model.train()
    start_time = time.time()

    for step in range(1, train_cfg.max_steps + 1):
        x, y = get_batch("train", train_cfg.batch_size, train_cfg.block_size, device=str(device))
        
        outputs = model(x, labels=y)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0:
            lr = scheduler.get_last_lr()[0]
            aux_loss = outputs["aux_loss"].item()
            print(f"Step {step:4d}/{train_cfg.max_steps} | Loss: {loss.item():.4f} | MoE Aux: {aux_loss:.4f} | LR: {lr:.2e}", flush=True)

        # Estimate validation loss and sample output text
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            losses = estimate_loss(model, train_cfg, device)
            print(f"Validation loss: train {losses['train']:.4f}, val {losses['val']:.4f}", flush=True)

            model.eval()
            prompt = "\n"
            prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8)
            print(f"Generated text sample:\n{decode(gen_ids[0].tolist())}\n", flush=True)
            model.train()

    print(f"Training completed in {time.time() - start_time:.1f} seconds", flush=True)


if __name__ == "__main__":
    train()
