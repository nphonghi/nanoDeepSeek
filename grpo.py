import copy
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig
from model import NanoDeepSeek
from data import vocab_size, encode, decode


def reward_function(text: str) -> float:
    # Base score initialization
    score = 0.0

    # Reward dialogue structure
    if ":" in text:
        score += 1.5
    if "\n" in text:
        score += 1.0

    # Reward reasonable output length
    if 10 <= len(text) <= 50:
        score += 1.0
    return score


def grpo_demo():
    # Hardware device detection
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Running GRPO demo on device: {device}")

    # Model architecture configuration
    model_cfg = ModelConfig(vocab_size=vocab_size)

    # Policy model to optimize and frozen reference model for KL divergence penalty
    policy_model = NanoDeepSeek(model_cfg).to(device)

    # Load initial weights from pretrained base model or fine-tuned model if available
    base_dir = Path(__file__).resolve().parent
    init_checkpoint = None
    for ckpt_name in ["lora_model.pt", "base_model.pt"]:
        if (base_dir / ckpt_name).exists():
            init_checkpoint = base_dir / ckpt_name
            break

    if init_checkpoint is not None:
        state_dict = torch.load(init_checkpoint, map_location=device)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if "lora_A" in k or "lora_B" in k:
                continue
            new_k = k.replace(".base_layer.weight", ".weight")
            cleaned_state_dict[new_k] = v
        policy_model.load_state_dict(cleaned_state_dict)
        if init_checkpoint.name == "lora_model.pt":
            print(f"Loaded fine-tuned model weights from {init_checkpoint.name} (LoRA -> GRPO)")
        else:
            print(f"Loaded base model weights from {init_checkpoint.name} (lora_model.pt not found, running Base -> GRPO)")
    else:
        print("Warning: Neither lora_model.pt nor base_model.pt found, starting from random initialization")

    ref_model = copy.deepcopy(policy_model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer and training hyperparameters
    optimizer = AdamW(policy_model.parameters(), lr=5e-5)
    group_size = 4
    kl_coef = 0.05

    prompt = "LORD:\n"
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    print(f"Training prompt: {repr(prompt)}")
    print(f"Group size G: {group_size}\n")

    # Reinforcement learning training loop (GRPO)
    policy_model.train()
    clip_eps = 0.2
    prompt_len = prompt_ids.size(1)

    for step in range(1, 11):
        responses = []
        response_texts = []
        rewards = []

        sys.stdout.write(f"\rGRPO Step {step:2d}/10 | Generating {group_size} rollouts...")
        sys.stdout.flush()

        # Rollout: Generate a group of G responses for the exact same prompt
        with torch.no_grad():
            for _ in range(group_size):
                gen = policy_model.generate(prompt_ids, max_new_tokens=25, temperature=1.0)
                resp_text = decode(gen[0].tolist())
                responses.append(gen)
                response_texts.append(resp_text)
                rewards.append(reward_function(resp_text))

        # Group Relative Advantage: Normalize rewards across group G
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # Compute DeepSeek-R1 GRPO loss (token-level policy gradient + clipping + KL penalty)
        total_loss = torch.tensor(0.0, device=device)

        for i in range(group_size):
            full_ids = responses[i]
            shift_labels = full_ids[:, 1:].contiguous()

            # Forward current policy model
            policy_outputs = policy_model(full_ids)
            policy_logits = policy_outputs["logits"][:, :-1, :].contiguous()
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            token_log_probs = policy_log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Forward frozen reference model
            with torch.no_grad():
                ref_outputs = ref_model(full_ids)
                ref_logits = ref_outputs["logits"][:, :-1, :].contiguous()
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_token_log_probs = ref_log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mask out prompt tokens so loss ONLY optimizes generated completion tokens!
            completion_mask = torch.zeros_like(token_log_probs)
            completion_mask[:, (prompt_len - 1):] = 1.0

            # Detached rollout log_probs for PPO/GRPO importance sampling ratio
            old_log_probs = token_log_probs.detach()
            ratio = torch.exp(token_log_probs - old_log_probs)

            # Clipped surrogate objective
            adv = advantages[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2)

            # DeepSeek-R1 unbiased KL penalty: D_kl = exp(ref - pi) - (ref - pi) - 1
            log_diff = ref_token_log_probs - token_log_probs
            kl_penalty = torch.exp(log_diff) - log_diff - 1.0

            per_token_loss = policy_loss + kl_coef * kl_penalty
            sample_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            total_loss = total_loss + sample_loss

        # Backpropagation and optimizer step
        total_loss = total_loss / group_size

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        sys.stdout.write(f"\rGRPO Step {step:2d}/10 | Avg Reward: {mean_r.item():.2f} | Loss: {total_loss.item():.4f}\n")
        sys.stdout.flush()
        if step % 5 == 0:
            print(f"Sample response: {repr(response_texts[0][:40])} (Reward: {rewards[0]:.1f})", flush=True)

    print("\nGRPO training loop completed successfully")

    # Save DeepSeek-R1 aligned policy checkpoint
    r1_checkpoint = Path(__file__).resolve().parent / "r1_model.pt"
    torch.save(policy_model.state_dict(), r1_checkpoint)
    print(f"Saved DeepSeek-R1 aligned model checkpoint to {r1_checkpoint.name}")


if __name__ == "__main__":
    grpo_demo()
