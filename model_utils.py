import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import Optional, List, Tuple

device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("GPU not found! This script is GPU-only.")

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_base_model(
    model_name: str,
    use_4bit: bool = False,
    bnb_config: Optional[dict] = None,
    torch_dtype: Optional[torch.dtype] = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    tokenizer = load_tokenizer(model_name)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(bnb_config or {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        })

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map={"": device},
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch_dtype
        )

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def apply_lora_adapters(
    model: AutoModelForCausalLM,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    return model


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} | All params: {total} | Trainable%: {100 * trainable / total:.4f}")


def save_peft_adapter(model: PeftModel, tokenizer, output_dir: str):
    import os
    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "peft_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved PEFT adapter to: {adapter_dir}, tokenizer to: {output_dir}")


def load_peft_adapter(base_model, adapter_dir: str):
    import os
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"{adapter_dir} not found")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.to(device)
    return peft_model
