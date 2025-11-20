import os
import json
import torch
import random
import logging

from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator
)

from data_utils import load_text_dataset, get_tokenizer, tokenize_and_group
from model_utils import (
    load_base_model,
    apply_lora_adapters,
    print_trainable_parameters,
    save_peft_adapter,
    move_model_to_gpu
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(ds, tokenizer, max_length):
    tokenized = {}
    for split_name, split_data in ds.items():
        tokenized[split_name] = split_data.map(
            lambda x: tokenize_and_group(x, tokenizer, max_length),
            batched=True,
            remove_columns=split_data.column_names,
        )
    return tokenized


def main(config_path: str):
    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))

    # Load dataset
    ds = load_text_dataset(
        train_file=cfg.get("train_file"),
        valid_file=cfg.get("valid_file"),
        dataset_name=cfg.get("dataset_name"),
    )

    # Load model + tokenizer (NO device_map here)
    use_4bit = cfg.get("use_4bit_quantization", False)
    model, tokenizer = load_base_model(
        cfg["model_name"],
        use_4bit=use_4bit,
        bnb_config=None,
        torch_dtype=torch.float16 if cfg.get("fp16", True) else None,
    )

    # Apply LoRA
    if cfg.get("use_lora", True):
        model = apply_lora_adapters(
            model,
            r=cfg.get("lora_rank", 8),
            alpha=cfg.get("lora_alpha", 16),
            dropout=cfg.get("lora_dropout", 0.1),
            target_modules=cfg.get("target_modules")
        )
        print_trainable_parameters(model)


    model = move_model_to_gpu(model)
    print("Model device →", next(model.parameters()).device)

    # Tokenize dataset
    tokenized = prepare_dataset(
        ds,
        tokenizer,
        max_length=cfg.get("max_seq_length", 128)
    )

    # Trainer setup
    output_dir = cfg.get("output_dir", "outputs")
    logging_dir = cfg.get("logging_dir", "outputs/logs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_steps=cfg.get("warmup_steps", 100),
        weight_decay=cfg.get("weight_decay", 0.0),
        fp16=cfg.get("fp16", True),
        save_steps=cfg.get("save_steps", 500),
        logging_steps=cfg.get("logging_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=cfg.get("save_steps", 500),
        report_to="tensorboard",
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None),
        data_collator=default_data_collator,
    )

    # Train
    trainer.train()

    # Save adapters
    save_peft_adapter(model, tokenizer, output_dir)

    log.info("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)