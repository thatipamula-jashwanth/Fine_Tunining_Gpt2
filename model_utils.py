import os
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from typing import Optional, List, Tuple



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
):

    tokenizer = load_tokenizer(model_name)


    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise ImportError(
                "bitsandbytes not installed but use_4bit=True. Error: " + str(e)
            )

        bnb_cfg = BitsAndBytesConfig((bnb_config or {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        }))

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=None,                 
            torch_dtype=torch.float16,
        )

        # Required for LoRA on quantized models
        model = prepare_model_for_kbit_training(model)


    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,                 
            torch_dtype=torch_dtype
        )

    # If tokenizer expanded, resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer



def apply_lora_adapters(
    model: AutoModelForCausalLM,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
):
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    return model



def move_model_to_gpu(model):
  
    if torch.cuda.is_available():
        model.to("cuda")
    return model


#
def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable} | Total: {total} | {100 * trainable / total:.4f}%")



def save_peft_adapter(model: PeftModel, tokenizer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "peft_adapter")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved adapter to: {adapter_dir}")
    print(f"Saved tokenizer to: {output_dir}")


def load_peft_adapter(base_model, adapter_dir: str):
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"{adapter_dir} not found!")
    return PeftModel.from_pretrained(base_model, adapter_dir)