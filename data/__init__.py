import os
import pickle
from pathlib import Path
from typing import Tuple, List
import torch

from .prepare import prepare

DATA_DIR = Path(__file__).resolve().parent
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"

# Prepare binary token files if not present
if not TRAIN_BIN_PATH.exists() or not VAL_BIN_PATH.exists() or not META_PATH.exists():
    prepare()

# Load vocabulary metadata
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vocab_size: int = meta["vocab_size"]
stoi: dict = meta["stoi"]
itos: dict = meta["itos"]


def encode(s: str) -> List[int]:
    return [stoi[c] for c in s if c in stoi]


def decode(l: List[int]) -> str:
    return "".join([itos[i] for i in l if i in itos])


# Memory-map binary token files on demand
try:
    import numpy as np
    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint16, mode="r")

    def get_batch(
        split: str,
        batch_size: int = 32,
        block_size: int = 64,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((source[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((source[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

except ImportError:
    # Pure PyTorch fallback when numpy is not available
    with open(TRAIN_BIN_PATH, "rb") as f:
        train_data = torch.frombuffer(f.read(), dtype=torch.int16).long()
    with open(VAL_BIN_PATH, "rb") as f:
        val_data = torch.frombuffer(f.read(), dtype=torch.int16).long()

    def get_batch(
        split: str,
        batch_size: int = 32,
        block_size: int = 64,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - block_size, (batch_size,))
        x = torch.stack([source[i : i + block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)

