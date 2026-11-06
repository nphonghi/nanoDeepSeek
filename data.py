import os
import pickle
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"

# Run data preparation if binary files or metadata are missing
if not TRAIN_BIN_PATH.exists() or not VAL_BIN_PATH.exists() or not META_PATH.exists():
    import sys
    sys.path.insert(0, str(DATA_DIR))
    from prepare import prepare
    prepare()

# Load vocabulary metadata matching nanoGPT
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vocab_size: int = meta["vocab_size"]
stoi: dict = meta["stoi"]
itos: dict = meta["itos"]


def encode(s: str) -> List[int]:
    return [stoi[c] for c in s if c in stoi]


def decode(l: List[int]) -> str:
    return "".join([itos[i] for i in l if i in itos])


def get_batch(
    split: str,
    batch_size: int = 32,
    block_size: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Memory-map binary token files on demand to match nanoGPT
    bin_path = TRAIN_BIN_PATH if split == "train" else VAL_BIN_PATH
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)
