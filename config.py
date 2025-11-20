CONFIG = {
    "model_name": "gpt2",
    "dataset_name": None,
    "train_file": "data/train.txt",
    "valid_file": "data/valid.txt",

    "output_dir": "outputs",
    "logging_dir": "outputs/logs",

    "max_seq_length": 128,
    "train_on_inputs": True,

    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,

    "num_train_epochs": 3,
    "learning_rate": 0.0002,
    "weight_decay": 0.0,
    "warmup_steps": 100,

    "save_steps": 500,
    "logging_steps": 50,
    "save_total_limit": 3,

    "seed": 42,
    "fp16": True,

    "use_lora": True,
    "use_qlora": False,

    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,

    "target_modules": ["q_proj", "v_proj"],

    "use_4bit_quantization": False,
    "bnb_compute_dtype": "float16",

    # Force GPU usage explicitly
    "device": "cuda"
}
