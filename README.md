# Finetuning environment for BART summarization

This project contains utilities to create a Python virtual environment and install packages needed to finetune BART-like models for text summarization.

Quick start

1. From the project root run:

```bash
./setup_venv.sh
```

2. Activate the venv:

```bash
source .venv/bin/activate
```

3. (Optional) Register a Jupyter kernel:

```bash
python -m ipykernel install --user --name=bart-finetune --display-name "Python (bart-finetune)"
```

Notes
- If you have a CUDA-enabled GPU and want memory-efficient 8-bit training, consider installing `bitsandbytes` and a matching `bitsandbytes` build for your CUDA version.
- If you plan to use distributed or accelerated training, configure `accelerate` using `accelerate config`.
# LLM_Finetuning_SOAP
