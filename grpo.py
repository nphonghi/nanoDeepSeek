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
    # Reward responses having dialogue formatting and reasonable length
    score = 0.0
    if ":" in text:
        score += 1.5
    if "\n" in text:
        score += 1.0
    if 10 <= len(text) <= 50:
        score += 1.0
    return score


def grpo_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Running GRPO demo on device: {device}")

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

    # Policy model to optimize and frozen reference model for KL divergence penalty
    policy_model = NanoDeepSeek(model_cfg).to(device)
    ref_model = copy.deepcopy(policy_model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(policy_model.parameters(), lr=5e-5)
    group_size = 4
    kl_coef = 0.05

    prompt = "LORD:\n"
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    print(f"Training prompt: {repr(prompt)}")
    print(f"Group size G: {group_size}\n")

    policy_model.train()
    for step in range(1, 11):
        responses = []
        response_texts = []
        rewards = []

        # Generate a group of G responses for the same prompt
        with torch.no_grad():
            for _ in range(group_size):
                gen = policy_model.generate(prompt_ids, max_new_tokens=25, temperature=1.0)
                resp_text = decode(gen[0].tolist())
                responses.append(gen)
                response_texts.append(resp_text)
                rewards.append(reward_function(resp_text))

        # Normalize rewards within group to compute relative advantages
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # Compute policy gradient loss with advantage and KL penalty
        total_loss = torch.tensor(0.0, device=device)
        for i in range(group_size):
            full_ids = responses[i]
            policy_outputs = policy_model(full_ids, labels=full_ids)
            log_prob = -policy_outputs["loss"]

            with torch.no_grad():
                ref_outputs = ref_model(full_ids, labels=full_ids)
                ref_log_prob = -ref_outputs["loss"]

            kl = log_prob - ref_log_prob
            step_loss = -advantages[i] * log_prob + kl_coef * kl
            total_loss = total_loss + step_loss

        total_loss = total_loss / group_size

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"GRPO Step {step:2d} | Avg Reward: {mean_r.item():.2f} | Loss: {total_loss.item():.4f}")
        if step % 5 == 0:
            print(f"Sample response: {repr(response_texts[0][:40])} (Reward: {rewards[0]:.1f})")

    print("\nGRPO training loop completed successfully")


if __name__ == "__main__":
    grpo_demo()
