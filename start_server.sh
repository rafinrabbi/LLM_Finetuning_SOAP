#!/bin/bash

# Medical SOAP Summarization - Unified Server Startup Script
# ===========================================================

echo "🚀 Starting Medical SOAP Summarization Server..."
echo "=================================================="
echo ""

# Parse command line arguments
WITH_GRADIO=false
if [[ "$1" == "--with-gradio" ]] || [[ "$1" == "-g" ]]; then
    WITH_GRADIO=true
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if ! python -c "import gradio, flask, transformers" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements_flask.txt
fi

# Check if model exists
if [ ! -d "bart-large-cnn-finetuned-soap-model" ]; then
    echo ""
    echo "⚠️  WARNING: Model directory not found!"
    echo "    Expected location: ./bart-large-cnn-finetuned-soap-model"
    echo ""
    echo "    Please ensure your fine-tuned model is in the project directory."
    echo "    The server will start but may use the base model as fallback."
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
fi

echo ""
if [ "$WITH_GRADIO" = true ]; then
    echo "✅ Starting integrated Flask + Gradio server..."
    echo ""
    echo "📍 Server will be available at:"
    echo "   - Flask API:  http://localhost:5000/"
    echo "   - Gradio UI:  http://localhost:7860/"
    echo ""
    echo "Press Ctrl+C to stop both servers"
    echo "=================================================="
    echo ""
    
    # Start with Gradio
    python flask_api.py --with-gradio
else
    echo "✅ Starting Flask API server only..."
    echo ""
    echo "📍 Server will be available at:"
    echo "   - API:  http://localhost:5000/"
    echo ""
    echo "💡 To also start Gradio UI:"
    echo "   ./start_server.sh --with-gradio"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=================================================="
    echo ""
    
    # Start Flask only
    python flask_api.py
fi
