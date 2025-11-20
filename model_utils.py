import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import Optional, List, Tuple, Dict


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    return tokenizer


def load_base_model(
    model_name: str,
    use_4bit: bool = False,
    bnb_config: Optional[dict] = None,
    device_map: Optional[dict | str] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    tokenizer = load_tokenizer(model_name)

    if use_4bit:
        # Lazy import to keep errors clear if bnb is missing
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise ImportError("To use QLoRA you must install a compatible transformers + bitsandbytes. " +
                              "Error: " + str(e))

        bnb_cfg = BitsAndBytesConfig((bnb_config or {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        }))

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            torch_dtype=torch.float16,  # compute dtype for kernels
        )

        # Prepare model for k-bit training: this alters some internals so PEFT works with k-bit
        model = prepare_model_for_kbit_training(model)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )

    # Resize embeddings if tokenizer expanded (pad token)
    model.resize_token_embeddings(len(tokenizer))

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
    return model


def print_trainable_parameters(model):

    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:    #check if it will change during training
            trainable += num
    print(f"Trainable params: {trainable} | All params: {total} | Trainable%: {100 * trainable / total:.4f}")


def save_peft_adapter(model: PeftModel, tokenizer, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "peft_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved PEFT adapter to: {adapter_dir}, tokenizer to: {output_dir}")


def load_peft_adapter(base_model, adapter_dir: str):

    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"{adapter_dir} not found")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    return peft_model
