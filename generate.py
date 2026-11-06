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
    """
    Speculative decoding using DeepSeek-V3 MTP depth module as a built-in draft model.
    Proposes candidate token t+2 from (h_t, token_{t+1}) and verifies with KV cache.
    """
    model.eval()
    generated = input_ids.clone()
    tokens_generated = 0
    speculative_hits = 0

    # Prefill prompt and initialize compressed MLA KV cache
    outputs = model(generated, use_cache=True)
    past_key_values = outputs["past_key_values"]
    curr_logits = outputs["logits"][:, -1, :]
    curr_hidden = outputs["hidden_states"][:, -1:, :]

    # Speculative decoding loop
    while tokens_generated < max_new_tokens:
        next_token = torch.argmax(curr_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        tokens_generated += 1

        if tokens_generated >= max_new_tokens:
            break

        # Draft next candidate token t+2 using MTP Depth 1 module
        draft_token = None
        if model.mtp_head is not None:
            draft_token = model.mtp_head.draft_next_token(
                h_last=curr_hidden,
                next_token=next_token,
                embed_tokens=model.embed_tokens,
                lm_head=model.lm_head,
            )

        # Forward generated token through main model with compressed KV cache
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs["past_key_values"]
        curr_logits = outputs["logits"][:, -1, :]
        curr_hidden = outputs["hidden_states"][:, -1:, :]
        true_next_token = torch.argmax(curr_logits, dim=-1, keepdim=True)

        # Verify draft: if main model agrees with draft, accept both tokens
        if draft_token is not None and true_next_token.item() == draft_token.item():
            generated = torch.cat([generated, draft_token], dim=1)
            tokens_generated += 1
            speculative_hits += 1

            if tokens_generated >= max_new_tokens:
                break

            outputs = model(draft_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs["past_key_values"]
            curr_logits = outputs["logits"][:, -1, :]
            curr_hidden = outputs["hidden_states"][:, -1:, :]

    hit_rate = (speculative_hits / max(1, tokens_generated)) * 100
    print(f"MTP Speculative hit rate: {hit_rate:.1f}% ({speculative_hits} hits in {tokens_generated} tokens)")
    return generated


def main():
    # Detect hardware accelerator device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Running generation demo on device: {device}")

    # Instantiate NanoDeepSeek model configuration
    model_cfg = ModelConfig(vocab_size=vocab_size)
    model = NanoDeepSeek(model_cfg).to(device)

    # Load latest available model checkpoint in the pipeline: R1 > LoRA > Base
    base_dir = Path(__file__).resolve().parent
    checkpoint_found = None
    for ckpt_name in ["r1_model.pt", "lora_model.pt", "base_model.pt"]:
        if (base_dir / ckpt_name).exists():
            checkpoint_found = base_dir / ckpt_name
            break

    if checkpoint_found is not None:
        state_dict = torch.load(checkpoint_found, map_location=device)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if "lora_A" in k or "lora_B" in k:
                continue
            new_k = k.replace(".base_layer.weight", ".weight")
            cleaned_state_dict[new_k] = v
        model.load_state_dict(cleaned_state_dict)
        print(f"Loaded model checkpoint from {checkpoint_found.name}")
    else:
        print("No trained checkpoint found, running with randomly initialized weights")

    model.eval()

    prompt = "KING:\nWhat is the"
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    print(f"Prompt: {repr(prompt)}")

    # Benchmark standard autoregressive generation
    start = time.time()
    out_std = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    time_std = time.time() - start
    print(f"\nStandard generation ({time_std:.3f}s):\n{decode(out_std[0].tolist())}")

    # Benchmark MTP speculative decoding
    start = time.time()
    out_spec = speculative_generate_mtp(model, prompt_ids, max_new_tokens=40)
    time_spec = time.time() - start
    print(f"\nSpeculative generation ({time_spec:.3f}s):\n{decode(out_spec[0].tolist())}")


if __name__ == "__main__":
    main()
