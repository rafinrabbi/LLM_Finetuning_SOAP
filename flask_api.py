"""
Medical SOAP Summarization - Flask REST API
===========================================

A production-ready Flask API for generating SOAP notes from medical dialogues.

Author: [Your Name]
Contact: [Your Email]
Date: October 2025

Endpoints:
    GET  /              - API information
    GET  /health        - Health check
    POST /generate      - Generate SOAP note
    POST /batch         - Batch generation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import os
import time
from typing import Dict, List, Any
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "./bart-large-cnn-finetuned-soap-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MODEL LOADING
# ============================================================================

logger.info(f"Loading model from {MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Load generation config if available
    try:
        generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    except:
        generation_config = None
        logger.warning("Generation config not found, using defaults")
    
    logger.info(f"✅ Model loaded successfully on {DEVICE}")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_soap_note(
    dialogue: str,
    max_length: int = 900,
    min_length: int = 150,
    num_beams: int = 4,
    length_penalty: float = 1.5,
    repetition_penalty: float = 1.2
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
            if generation_config:
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
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
        logger.error(f"Error generating summary: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "name": "Medical SOAP Summarization API",
        "version": "1.0.0",
        "description": "Generate SOAP notes from medical dialogues using fine-tuned BART",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/generate": "Generate SOAP note (POST)",
            "/batch": "Batch generation (POST)"
        },
        "model": {
            "name": "BART-Large-CNN (Fine-tuned)",
            "device": DEVICE,
            "status": "ready"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "timestamp": time.time()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate SOAP note from medical dialogue.
    
    Request body:
        {
            "dialogue": "Medical dialogue text...",
            "max_length": 900 (optional),
            "min_length": 150 (optional),
            "num_beams": 4 (optional),
            "length_penalty": 1.5 (optional),
            "repetition_penalty": 1.2 (optional)
        }
    
    Response:
        {
            "success": true,
            "soap_note": "Generated SOAP note...",
            "metadata": {...}
        }
    """
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'dialogue' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'dialogue' field in request body"
            }), 400
        
        dialogue = data['dialogue']
        
        if not dialogue or len(dialogue.strip()) == 0:
            return jsonify({
                "success": False,
                "error": "Empty dialogue provided"
            }), 400
        
        # Get optional parameters
        max_length = data.get('max_length', 900)
        min_length = data.get('min_length', 150)
        num_beams = data.get('num_beams', 4)
        length_penalty = data.get('length_penalty', 1.5)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        
        # Validate parameters
        if not (200 <= max_length <= 1024):
            return jsonify({
                "success": False,
                "error": "max_length must be between 200 and 1024"
            }), 400
        
        if not (0.5 <= length_penalty <= 2.0):
            return jsonify({
                "success": False,
                "error": "length_penalty must be between 0.5 and 2.0"
            }), 400
        
        # Generate SOAP note
        result = generate_soap_note(
            dialogue=dialogue,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in /generate endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def batch_generate():
    """
    Generate SOAP notes for multiple dialogues.
    
    Request body:
        {
            "dialogues": ["dialogue1", "dialogue2", ...],
            "max_length": 900 (optional),
            "min_length": 150 (optional),
            "num_beams": 4 (optional),
            "length_penalty": 1.5 (optional),
            "repetition_penalty": 1.2 (optional)
        }
    
    Response:
        {
            "success": true,
            "results": [
                {"soap_note": "...", "metadata": {...}},
                ...
            ],
            "total_processed": 2,
            "total_time_seconds": 1.234
        }
    """
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'dialogues' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'dialogues' field in request body"
            }), 400
        
        dialogues = data['dialogues']
        
        if not isinstance(dialogues, list) or len(dialogues) == 0:
            return jsonify({
                "success": False,
                "error": "dialogues must be a non-empty list"
            }), 400
        
        if len(dialogues) > 100:
            return jsonify({
                "success": False,
                "error": "Maximum 100 dialogues per batch request"
            }), 400
        
        # Get optional parameters
        max_length = data.get('max_length', 900)
        min_length = data.get('min_length', 150)
        num_beams = data.get('num_beams', 4)
        length_penalty = data.get('length_penalty', 1.5)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        
        # Process batch
        start_time = time.time()
        results = []
        
        for dialogue in dialogues:
            result = generate_soap_note(
                dialogue=dialogue,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "results": results,
            "total_processed": len(results),
            "total_time_seconds": round(total_time, 3)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /batch endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False in production
    )
