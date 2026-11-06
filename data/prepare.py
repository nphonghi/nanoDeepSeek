import os
import pickle
import urllib.request
import array
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"
DATA_URL = "https://huggingface.co/datasets/tiny_shakespeare/raw/main/tiny_shakespeare.txt"


def prepare():
    # Fetch corpus directly into memory without saving raw text to disk
    print(f"Downloading Tiny Shakespeare corpus into memory from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as response:
        data = response.read().decode("utf-8")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(l):
        return "".join([itos[i] for i in l if i in itos])

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = array.array("H", encode(train_data))
    val_ids = array.array("H", encode(val_data))

    # Export pre-tokenized binary files directly
    with open(TRAIN_BIN_PATH, "wb") as f:
        train_ids.tofile(f)
    with open(VAL_BIN_PATH, "wb") as f:
        val_ids.tofile(f)

    # Save vocabulary metadata mapping characters to integers
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Prepared train.bin ({len(train_ids):,} tokens) and val.bin ({len(val_ids):,} tokens)")


if __name__ == "__main__":
    prepare()
