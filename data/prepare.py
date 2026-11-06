import os
import pickle
import urllib.request
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATA_DIR / "input.txt"
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare():
    # Download Tiny Shakespeare dataset via Python if not already present
    if not INPUT_PATH.exists():
        print(f"Downloading Tiny Shakespeare from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, INPUT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = f.read()

    print(f"Length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(l):
        return "".join([itos[i] for i in l if i in itos])

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    print(f"Train split tokens: {len(train_ids):,}")
    print(f"Val split tokens: {len(val_ids):,}")

    # Export binary token files for fast memmap loading
    train_ids.tofile(TRAIN_BIN_PATH)
    val_ids.tofile(VAL_BIN_PATH)

    # Save vocabulary metadata mapping characters to integers
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print("Data preparation completed successfully")


if __name__ == "__main__":
    prepare()

