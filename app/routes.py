# app/routes.py
"""
Flask API routes for Medical SOAP Summarization
================================================
Defines all REST API endpoints and error handlers.
"""

from flask import Blueprint, request, jsonify
import time
from typing import List
from .service import generate_soap_note, get_model_info
from .config import MAX_BATCH_SIZE
import logging

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# ============================================================================
# API ROUTES
# ============================================================================

@api_bp.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    model_info = get_model_info()
    
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
            "device": model_info["device"],
            "status": "ready" if model_info["model_loaded"] else "not loaded"
        }
    })

@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_info = get_model_info()
    
    return jsonify({
        "status": "healthy" if model_info["model_loaded"] else "unhealthy",
        "model_loaded": model_info["model_loaded"],
        "device": model_info["device"],
        "timestamp": time.time()
    })

@api_bp.route('/generate', methods=['POST'])
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

@api_bp.route('/batch', methods=['POST'])
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
        
        if len(dialogues) > MAX_BATCH_SIZE:
            return jsonify({
                "success": False,
                "error": f"Maximum {MAX_BATCH_SIZE} dialogues per batch request"
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

def register_error_handlers(app):
    """Register error handlers to the Flask app"""
    
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
