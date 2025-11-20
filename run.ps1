# Optional: reduce extra warnings from HuggingFace
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"

# Optional but recommended: disable tokenizer parallelism
$env:TOKENIZERS_PARALLELISM = "false"

# GPU selection (optional for Windows — bitsandbytes usually CPU-only)
# $env:CUDA_VISIBLE_DEVICES = "0"

# Start training
python train.py --config config.json

#running
#-----
# ---> powershell -ExecutionPolicy Bypass -File run.ps1
