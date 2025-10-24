# wsgi.py
"""
WSGI entry point for Flask-only server
=======================================
Run with: python wsgi.py
Or with Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
"""

import os
import logging
from app import create_app
from app.config import FLASK_HOST, FLASK_PORT

# Create Flask application
app = create_app()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info(f"Starting Flask API server on {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"API endpoints available at: http://localhost:{FLASK_PORT}/")
    logger.info("")
    logger.info("💡 Tip: To also start Gradio UI, run:")
    logger.info("   python gradio_main.py (in another terminal)")
    logger.info("   or")
    logger.info("   python main.py --with-gradio (integrated mode)")
    logger.info("")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False
    )
