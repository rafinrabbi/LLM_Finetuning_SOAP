FROM python:3.11-slim

# Set non-interactive and unbuffered Python output for clearer logs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_home \
    # Default model repo (can be overridden in Spaces Secrets / env)
    MODEL_PATH=rawhaturrafin/Finetuned_BART_large_CNN_for_SOAP_Summary \
    HF_MODEL_SUBFOLDER=saved-finetuned-model

WORKDIR /app

# Install system dependencies required for some Python packages and HF Hub
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first to leverage layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose typical ports used by the app (Gradio + Flask)
EXPOSE 7860 5000

# Default command: run integrated Flask + Gradio UI
# In Hugging Face Spaces you can override env vars (MODEL_PATH, HUGGINGFACE_HUB_TOKEN) via Settings -> Secrets
CMD ["python", "main.py", "--with-gradio"]
# CMD ["python", "main.py"]
