# main.py
"""
Unified entry point for Flask + Gradio server
==============================================
Run Flask only:           python main.py
Run Flask + Gradio:       python main.py --with-gradio
"""

import sys
import logging
import threading
from app import create_app
from app.gradio_ui import create_gradio_interface
from app.config import FLASK_HOST, FLASK_PORT, GRADIO_PORT

logger = logging.getLogger(__name__)

def run_flask(app):
    """Run Flask server"""
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        use_reloader=False
    )

def run_gradio(demo):
    """Run Gradio server"""
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True
    )

if __name__ == '__main__':
    # Check for --with-gradio flag
    with_gradio = '--with-gradio' in sys.argv or '-g' in sys.argv
    
    # Create Flask app
    app = create_app()
    
    if with_gradio:
        logger.info("=" * 70)
        logger.info("INTEGRATED MODE: Flask API + Gradio UI")
        logger.info("=" * 70)
        logger.info(f"Flask API will be available at: http://localhost:{FLASK_PORT}/")
        logger.info(f"Gradio UI will be available at: http://localhost:{GRADIO_PORT}/")
        logger.info("")
        logger.info("Press Ctrl+C to stop both servers")
        logger.info("=" * 70)
        
        # Create Gradio demo
        gradio_demo = create_gradio_interface()
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask, args=(app,), daemon=True)
        flask_thread.start()
        
        logger.info("✅ Flask API started in background thread")
        
        # Run Gradio in main thread (blocks)
        logger.info("✅ Starting Gradio UI...")
        run_gradio(gradio_demo)
        
    else:
        logger.info("=" * 70)
        logger.info("FLASK API ONLY MODE")
        logger.info("=" * 70)
        logger.info(f"API will be available at: http://localhost:{FLASK_PORT}/")
        logger.info("")
        logger.info("💡 Tip: To also start Gradio UI, use:")
        logger.info("   python main.py --with-gradio")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 70)
        
        # Run Flask only
        run_flask(app)
