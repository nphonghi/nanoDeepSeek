import time
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig
from model import NanoDeepSeek
from data import vocab_size, encode, decode


@torch.no_grad()
def speculative_generate_mtp(
    model: NanoDeepSeek,
    input_ids: torch.Tensor,
    max_new_tokens: int = 40,
) -> torch.Tensor:
    # Use MTP head as a built-in draft model to guess ahead
    model.eval()
    generated = input_ids.clone()
    tokens_generated = 0
    speculative_hits = 0

    while tokens_generated < max_new_tokens:
        outputs = model(generated)
        main_logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(main_logits, dim=-1, keepdim=True)

        # Propose draft token from MTP auxiliary head
        if outputs["mtp_logits"] is not None and len(outputs["mtp_logits"]) > 0:
            draft_token = torch.argmax(outputs["mtp_logits"][0][:, -1, :], dim=-1, keepdim=True)

            # Verify both tokens in a single forward evaluation
            candidate = torch.cat([generated, next_token, draft_token], dim=1)
            verify_outputs = model(candidate[:, :-1])
            verified_token = torch.argmax(verify_outputs["logits"][:, -1, :], dim=-1, keepdim=True)

            if verified_token.item() == draft_token.item():
                generated = candidate
                tokens_generated += 2
                speculative_hits += 1
                continue

        generated = torch.cat([generated, next_token], dim=1)
        tokens_generated += 1

    hit_rate = (speculative_hits / max(1, tokens_generated)) * 100
    print(f"MTP Speculative hit rate: {hit_rate:.1f}%")
    return generated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Running generation demo on device: {device}")

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
    model.eval()

    prompt = "KING:\nWhat is the"
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    print(f"Prompt: {repr(prompt)}")

    start = time.time()
    out_std = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    time_std = time.time() - start
    print(f"\nStandard generation ({time_std:.3f}s):\n{decode(out_std[0].tolist())}")

    start = time.time()
    out_spec = speculative_generate_mtp(model, prompt_ids, max_new_tokens=40)
    time_spec = time.time() - start
    print(f"\nSpeculative generation ({time_spec:.3f}s):\n{decode(out_spec[0].tolist())}")


if __name__ == "__main__":
    main()
