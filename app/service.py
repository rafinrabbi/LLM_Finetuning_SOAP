# app/service.py
"""
Model service module for SOAP note generation
==============================================
Handles model loading and SOAP note generation logic.
"""

import time
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from .config import (
    MODEL_PATH, DEVICE,
    GEN_MAX_LENGTH, GEN_MIN_LENGTH, GEN_NUM_BEAMS,
    GEN_LENGTH_PENALTY, GEN_REPETITION_PENALTY
)
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLETON MODEL LOADING (Lazy loading to avoid double-loading in workers)
# ============================================================================

_tokenizer = None
_model = None
_generation_config: Optional[GenerationConfig] = None

def load_model_once():
    """Load model, tokenizer, and generation config once (singleton pattern)"""
    global _tokenizer, _model, _generation_config
    
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model, _generation_config

    logger.info(f"Loading model from {MODEL_PATH}...")
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        _model.to(DEVICE)
        _model.eval()

        # Load generation config if available
        try:
            _generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
        except Exception:
            _generation_config = None
            logger.warning("Generation config not found, using defaults")

        logger.info(f"✅ Model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    return _tokenizer, _model, _generation_config

# ============================================================================
# SOAP NOTE GENERATION
# ============================================================================

def generate_soap_note(
    dialogue: str,
    max_length: int = GEN_MAX_LENGTH,
    min_length: int = GEN_MIN_LENGTH,
    num_beams: int = GEN_NUM_BEAMS,
    length_penalty: float = GEN_LENGTH_PENALTY,
    repetition_penalty: float = GEN_REPETITION_PENALTY,
) -> Dict[str, Any]:
    """
    Generate SOAP note from medical dialogue.
    
    Args:
        dialogue: Medical dialogue text
        max_length: Maximum length of generated summary
        min_length: Minimum length of generated summary
        num_beams: Number of beams for beam search
        length_penalty: Length penalty for generation
        repetition_penalty: Repetition penalty
    
    Returns:
        Dictionary containing generated SOAP note and metadata
    """
    tokenizer, model, gen_config = load_model_once()
    start_time = time.time()

    try:
        # Prepare input
        prompt = "Summarize the following medical dialogue into a concise SOAP note:\n" + dialogue
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(DEVICE)

        # Generate
        with torch.no_grad():
            if gen_config:
                # Ensure decoder_start_token_id is set
                gen_config.decoder_start_token_id = model.config.decoder_start_token_id
                
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                )

        # Decode
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time

        return {
            "success": True,
            "soap_note": summary,
            "metadata": {
                "input_length": len(dialogue),
                "output_length": len(summary),
                "compression_ratio": round(len(summary) / len(dialogue), 3),
                "tokens_generated": len(outputs[0]),
                "generation_time_seconds": round(generation_time, 3),
                "device": DEVICE
            }
        }

    except Exception as e:
        logger.exception("Error generating summary")
        return {
            "success": False,
            "error": str(e)
        }

def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model"""
    tokenizer, model, gen_config = load_model_once()
    
    return {
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "generation_config_loaded": gen_config is not None,
    }
