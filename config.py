CONFIG = {
    "model_name": "gpt2",
    "dataset_name": "roneneldan/TinyStories",
    "train_file": None,
    "valid_file": None,

    "output_dir": "outputs",
    "logging_dir": "outputs/logs",

    "max_seq_length": 128,
    "train_on_inputs": True,


    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,

    "num_train_epochs": 1,
    "learning_rate": 0.0002,
    "weight_decay": 0.0,
    "warmup_steps": 50,

    "save_steps": 500,
    "logging_steps": 50,
    "save_total_limit": 3,

    "seed": 42,


    "fp16": False,

    "use_lora": True,
    "use_qlora": False,

    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,

    "target_modules": ["c_attn"],

    "use_4bit_quantization": False,
    "bnb_compute_dtype": "float32",


    "device_map": "cpu"
}
