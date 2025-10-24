# gradio_main.py
"""
Gradio-only server entry point
===============================
Run with: python gradio_main.py
"""

import logging
from app.gradio_ui import launch_gradio
from app.config import GRADIO_PORT

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info(f"Starting Gradio UI server on port {GRADIO_PORT}")
    logger.info(f"Gradio UI available at: http://localhost:{GRADIO_PORT}/")
    logger.info("")
    
    launch_gradio(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False
    )
