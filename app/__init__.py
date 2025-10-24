# app/__init__.py
"""
Flask Application Factory for Medical SOAP Summarization
=========================================================
Creates and configures the Flask application with all components.
"""

from flask import Flask
from flask_cors import CORS
import logging
from .config import LOG_LEVEL, LOG_FORMAT
from .routes import api_bp, register_error_handlers

def create_app():
    """
    Application factory function.
    Creates and configures the Flask app with all blueprints and extensions.
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    logger = logging.getLogger(__name__)
    
    # Create Flask app
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    logger.info("Initializing Medical SOAP Summarization API")
    
    # Register blueprints
    app.register_blueprint(api_bp)
    logger.info("✅ API routes registered")
    
    # Register error handlers
    register_error_handlers(app)
    logger.info("✅ Error handlers registered")
    
    # Pre-load model (lazy loading happens on first request)
    from .service import load_model_once
    try:
        load_model_once()
        logger.info("✅ Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Model will be loaded on first request: {e}")
    
    return app
