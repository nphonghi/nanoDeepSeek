# nanoDeepSeek

A minimalist, hackable implementation of the **DeepSeek-V3** and **DeepSeek-R1** architectures in pure PyTorch, inspired by Andrej Karpathy's `nanoGPT`.

All core architectural building blocks are separated into dedicated, modular files under `modules/`. The dataset uses the classic Tiny Shakespeare corpus matching nanoGPT.

---

## Directory Structure

```text
nanoDeepSeek/
├── config.py           # Configuration classes (ModelConfig, MoEConfig, MTPConfig, LoRAConfig)
├── modules/            # Separated architectural building blocks
│   ├── norm.py         # RMSNorm (Root Mean Square Normalization)
│   ├── rope.py         # Rotary Position Embedding (RoPE)
│   ├── attention.py    # Multi-Head Latent Attention (MLA) with low-rank KV compression & decoupled RoPE
│   ├── moe.py          # DeepSeekMoE (Shared experts, routed experts, top-k gate, aux loss)
│   ├── mtp.py          # Multi-Token Prediction (MTP head for multi-step ahead prediction)
│   └── lora.py         # Native PyTorch LoRA adapter (LoRALinear, apply_lora, merge_weights)
├── model.py            # TransformerBlock assembly and full NanoDeepSeek model
├── data.py             # Memory-mapped data loader (get_batch, encode, decode)
├── train.py            # Pretraining script with validation loss estimation & text sampling
├── train_lora.py       # LoRA fine-tuning script freezing base weights and merging adapters
├── generate.py         # Text generation with MTP speculative decoding
├── grpo.py             # DeepSeek-R1 Group Relative Policy Optimization alignment demo
└── data/
    ├── prepare.py      # Python script to download & tokenize data into binary files
    ├── input.txt       # Raw Tiny Shakespeare corpus text (1.1 MB)
    ├── train.bin       # Binary encoded training tokens (np.uint16)
    ├── val.bin         # Binary encoded validation tokens (np.uint16)
    └── meta.pkl        # Character vocabulary mapping metadata
```

---

## DeepSeek Architectural Components

- **Multi-Head Latent Attention (MLA)** (`modules/attention.py`):
  Compresses keys and values into a shared low-rank latent vector $c_{kv}$, reducing KV cache memory footprint by 5-8x during generation. Uses decoupled rotary embeddings ($q^{nope}$ and $q^{rope}$) so position information is preserved without expanding KV cache size.
  
- **DeepSeekMoE** (`modules/moe.py`):
  Combines dedicated shared experts (always active across all tokens for universal knowledge) with top-$k$ routed experts (activated on demand). Includes auxiliary load-balancing loss to ensure uniform expert utilization.

- **Multi-Token Prediction (MTP)** (`modules/mtp.py`):
  Predicts future tokens $t+1, t+2$ concurrently from intermediate hidden states. Used during inference for speculative decoding (drafting and verifying 2 tokens per forward pass).

- **Native LoRA Adapter** (`modules/lora.py`):
  Implements $W = W_0 + \frac{\alpha}{r} (B \cdot A)$ in pure PyTorch. Targets MLA projection matrices (`q_down_proj`, `kv_down_proj`, `o_proj`), training less than 2% of total parameters, with one-click weight merging.

- **GRPO (Group Relative Policy Optimization)** (`grpo.py`):
  The core reinforcement learning algorithm behind DeepSeek-R1. Samples a group of $G$ outputs per prompt and normalizes rewards relative to the group mean and standard deviation, updating policy weights without requiring a separate critic model.

---

## Quick Start

### Prepare Dataset (nanoGPT style)
```bash
python3 nanoDeepSeek/data/prepare.py
```
*(Note: If omitted, `data.py` automatically downloads and prepares the data on first run.)*

### Pretrain on Tiny Shakespeare
```bash
python3 nanoDeepSeek/train.py
```

### Fine-Tune with LoRA
```bash
python3 nanoDeepSeek/train_lora.py
```

### Run Generation & MTP Speculative Decoding
```bash
python3 nanoDeepSeek/generate.py
```

### Run DeepSeek-R1 GRPO Alignment
```bash
python3 nanoDeepSeek/grpo.py
```
