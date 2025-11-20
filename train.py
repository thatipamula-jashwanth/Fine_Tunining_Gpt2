import os
import torch
import random
import logging
from transformers import TrainingArguments, Trainer, default_data_collator

from data_utils import load_text_dataset, get_tokenizer, tokenize_and_group
from model_utils import (
    load_base_model,
    apply_lora_adapters,
    print_trainable_parameters,
    save_peft_adapter
)
from config import CONFIG

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


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    ds = load_text_dataset(
        train_file=cfg["train_file"],
        valid_file=cfg["valid_file"],
        dataset_name=cfg["dataset_name"],
    )

    model, tokenizer = load_base_model(
        cfg["model_name"],
        use_4bit=cfg["use_4bit_quantization"],
        torch_dtype=torch.float16
    )

    if cfg["use_lora"]:
        model = apply_lora_adapters(
            model,
            r=cfg["lora_rank"],
            alpha=cfg["lora_alpha"],
            dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
        )
        print_trainable_parameters(model)

    tokenized = prepare_dataset(
        ds,
        tokenizer,
        cfg["max_seq_length"]
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        logging_dir=cfg["logging_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        fp16=True,
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=cfg["save_steps"],
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,  # already on CUDA
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None),
        data_collator=default_data_collator,
    )

    trainer.train()

    save_peft_adapter(model, tokenizer, cfg["output_dir"])
    log.info("Training complete.")


if __name__ == "__main__":
    main()
