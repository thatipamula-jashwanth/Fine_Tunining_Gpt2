from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import os

def load_text_dataset(train_file=None, valid_file=None, dataset_name=None):

    if dataset_name:
        ds = load_dataset(dataset_name)

        if "validation" not in ds and "train" in ds:
            ds = ds["train"].train_test_split(test_size=0.2, seed=42)
            ds = {
                "train": ds["train"],
                "validation": ds["test"]
            }
        return ds

    data = {}

    if train_file and os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        data["train"] = {"text": lines}

    if valid_file and os.path.exists(valid_file):
        with open(valid_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        data["validation"] = {"text": lines}

    if "train" in data and "validation" not in data:
        full = data["train"]["text"]
        split_idx = int(0.8 * len(full))

        data["train"] = {"text": full[:split_idx]}
        data["validation"] = {"text": full[split_idx:]}

    ds = {}
    for split, content in data.items():
        ds[split] = Dataset.from_dict(content)

    return ds


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def tokenize_and_group(examples, tokenizer, max_length):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    out["labels"] = out["input_ids"].copy()
    return out
