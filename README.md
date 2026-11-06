# nanoDeepSeek

A minimalist, hackable implementation of the **DeepSeek-V3** and **DeepSeek-R1** architectures in pure PyTorch.

All core architectural building blocks are separated into dedicated, modular files under `modules/`. The dataset uses the classic Tiny Shakespeare corpus.

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
├── data.py             # Memory-mapped binary token data loader
├── train.py            # Pretraining script with validation loss estimation & text sampling
├── train_lora.py       # LoRA fine-tuning script freezing base weights and merging adapters
├── generate.py         # Text generation with MTP speculative decoding
├── grpo.py             # DeepSeek-R1 Group Relative Policy Optimization alignment demo
└── data/
    ├── prepare.py      # Pre-tokenizes corpus into binary token files (uint16)
    ├── train.bin       # Binary encoded training tokens
    ├── val.bin         # Binary encoded validation tokens
    └── meta.pkl        # Character vocabulary mapping metadata
```

---

## DeepSeek Architectural Components

- **Multi-Head Latent Attention (MLA)** (`modules/attention.py`):
  Compresses keys and values into a shared low-rank latent vector $c_{kv}$, caching ONLY $c_t^{KV}$ and decoupled rotary key $k_t^R$ during generation (reducing KV cache memory footprint by 5-8x, and 56x at scale). Uses decoupled rotary embeddings ($q^{nope}, q^{rope}, k^{rope}$) and native `F.scaled_dot_product_attention`.
  
- **DeepSeekMoE** (`modules/moe.py`):
  Combines dedicated shared experts (always active across all tokens for universal knowledge) with top-$k$ routed experts. Supports both **DeepSeek-V2** (auxiliary load-balancing loss) and **DeepSeek-V3** (auxiliary-loss-free load balancing with dynamic expert bias $b_i$).

- **Multi-Token Prediction (MTP)** (`modules/mtp.py`):
  Faithful DeepSeek-V3 sequential depth module. Each MTP block combines representation $h_i^{k-1}$ with next token embedding $Emb(t_{i+k})$ and processes it through a dedicated Transformer block. Acts as a built-in draft model for KV-cached speculative decoding.

- **Native LoRA Adapter** (`modules/lora.py`):
  Implements $W = W_0 + \frac{\alpha}{r} (B \cdot A)$ in pure PyTorch. Targets MLA projection matrices (`q_down_proj`, `kv_down_proj`, `o_proj`), training less than 2% of total parameters, with one-click weight merging.

- **GRPO (Group Relative Policy Optimization)** (`grpo.py`):
  The core reinforcement learning algorithm behind DeepSeek-R1. Samples a group of $G$ outputs per prompt and normalizes rewards relative to the group. Features **prompt masking** (gradients only optimize generated completion tokens), **token-level PPO/GRPO clipping**, and DeepSeek-R1's **unbiased KL penalty** ($D_{KL} = e^{\Delta} - \Delta - 1$) without requiring a critic model.

---

## Sequential End-to-End Pipeline

The project runs as a connected sequential pipeline where each stage saves its checkpoint for the next stage:

```text
train.py (base_model.pt) ──> train_lora.py (lora_model.pt) ──> grpo.py (r1_model.pt) ──> generate.py
```

### 1. Pre-training Base Model
Trains the base NanoDeepSeek model and saves `base_model.pt`:
```bash
python3 train.py
```

### 2. LoRA Fine-Tuning
Loads `base_model.pt`, attaches low-rank adapters to MLA projections, fine-tunes, merges weights, and saves `lora_model.pt`:
```bash
python3 train_lora.py
```

### 3. DeepSeek-R1 GRPO Alignment
Loads pre-trained weights, creates frozen reference model, aligns policy via group relative advantages, and saves `r1_model.pt`:
```bash
python3 grpo.py
```

### 4. Inference & MTP Speculative Decoding
Automatically loads the latest available checkpoint (`r1_model.pt` > `lora_model.pt` > `base_model.pt`) and generates text:
```bash
python3 generate.py
```
