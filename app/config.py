# app/config.py
"""
Configuration module for Medical SOAP Summarization API
========================================================
Contains all configuration variables, paths, and constants.
"""

import os
import torch

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "./saved-finetuned-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# GENERATION PARAMETERS (Defaults)
# ============================================================================

GEN_MAX_LENGTH = 900
GEN_MIN_LENGTH = 150
GEN_NUM_BEAMS = 4
GEN_LENGTH_PENALTY = 1.5
GEN_REPETITION_PENALTY = 1.2

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", 5000))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", 7860))

# ============================================================================
# API CONFIGURATION
# ============================================================================

MAX_BATCH_SIZE = 100
API_TIMEOUT = 120

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
