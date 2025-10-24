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
import gradio as gr

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
                generation_config.decoder_start_token_id = model.config.decoder_start_token_id
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
# GRADIO UI INTEGRATION
# ============================================================================

# Example dialogues for Gradio UI
EXAMPLE_DIALOGUES = [
    """Patient: Hello doctor, I've been experiencing severe headaches for the past week, especially in the mornings.
Doctor: I see. Can you describe the type of pain you're experiencing?
Patient: It's a throbbing pain on both sides of my head. It gets worse when I move around.
Doctor: How long do these headaches typically last?
Patient: Usually 2-3 hours, but sometimes they persist throughout the day.
Doctor: Are you experiencing any other symptoms like nausea or sensitivity to light?
Patient: Yes, I feel nauseous sometimes and bright lights make it worse.
Doctor: Have you taken any medication for this?
Patient: I've tried over-the-counter pain relievers, but they don't seem to help much.
Doctor: Do you have a history of migraines in your family?
Patient: Yes, my mother suffers from migraines.
Doctor: Based on your symptoms, this appears to be migraines. I'll prescribe some medication and recommend lifestyle changes.""",
    
    """Patient: Doctor, I've had a persistent cough for about two weeks now.
Doctor: Is it a dry cough or are you producing mucus?
Patient: I'm coughing up yellowish mucus, especially in the morning.
Doctor: Do you have any fever or shortness of breath?
Patient: I had a low-grade fever last week, but it's gone now. I do feel a bit short of breath when climbing stairs.
Doctor: Any chest pain?
Patient: Yes, a bit of tightness in my chest when I cough.
Doctor: Let me listen to your lungs. You have some wheezing. This could be a respiratory infection.""",
    
    """Patient: I've been feeling extremely tired lately, even after getting enough sleep.
Doctor: How long has this been going on?
Patient: About a month now. I also feel dizzy when I stand up quickly.
Doctor: Any changes in your appetite or weight?
Patient: Yes, I've lost about 5 pounds without trying.
Doctor: Are you experiencing any other symptoms?
Patient: My skin looks paler than usual, and I've noticed I'm more irritable.
Doctor: We should run some blood tests. These symptoms could indicate anemia or thyroid issues."""
]

def gradio_generate_soap_note(dialogue: str, 
                               max_length: int = 900, 
                               min_length: int = 150,
                               num_beams: int = 4,
                               length_penalty: float = 1.5,
                               repetition_penalty: float = 1.2) -> str:
    """
    Gradio wrapper for generate_soap_note function.
    Returns formatted string instead of dict.
    """
    if not dialogue or len(dialogue.strip()) == 0:
        return "⚠️ Please enter a medical dialogue to summarize."
    
    try:
        result = generate_soap_note(
            dialogue=dialogue,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty
        )
        
        if result['success']:
            summary = result['soap_note']
            metadata = result['metadata']
            
            # Add statistics
            stats = f"\n\n📊 **Statistics:**\n"
            stats += f"- Input length: {metadata['input_length']} characters\n"
            stats += f"- Output length: {metadata['output_length']} characters\n"
            stats += f"- Compression ratio: {metadata['compression_ratio']*100:.1f}%\n"
            stats += f"- Tokens generated: {metadata['tokens_generated']} tokens\n"
            stats += f"- Generation time: {metadata['generation_time_seconds']} seconds\n"
            stats += f"- Device: {metadata['device']}"
            
            return summary + stats
        else:
            return f"❌ Error: {result['error']}"
            
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"

def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Medical SOAP Summarization") as demo:
        
        gr.Markdown("""
        # 🏥 Medical SOAP Note Generator
        
        Generate structured SOAP (Subjective, Objective, Assessment, Plan) notes from medical dialogues using AI.
        
        **Model:** Fine-tuned BART-Large-CNN for medical summarization
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Medical Dialogue")
                
                input_dialogue = gr.Textbox(
                    label="Medical Dialogue",
                    placeholder="Enter the medical dialogue between patient and doctor...",
                    lines=15,
                    max_lines=25
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    max_length_slider = gr.Slider(
                        minimum=200,
                        maximum=1024,
                        value=900,
                        step=50,
                        label="Max Length (tokens)",
                        info="Maximum length of generated summary"
                    )
                    
                    min_length_slider = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=150,
                        step=25,
                        label="Min Length (tokens)",
                        info="Minimum length of generated summary"
                    )
                    
                    num_beams_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of Beams",
                        info="Higher = better quality but slower"
                    )
                    
                    length_penalty_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.5,
                        step=0.1,
                        label="Length Penalty",
                        info="Encourages longer (>1.0) or shorter (<1.0) outputs"
                    )
                    
                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="Repetition Penalty",
                        info="Prevents repetitive text (higher = less repetition)"
                    )
                
                generate_btn = gr.Button("🔮 Generate SOAP Note", variant="primary", size="lg")
                
                gr.Markdown("### 📚 Example Dialogues")
                gr.Examples(
                    examples=EXAMPLE_DIALOGUES,
                    inputs=input_dialogue,
                    label="Click an example to load it"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Generated SOAP Note")
                
                output_soap = gr.Textbox(
                    label="SOAP Note",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
                
                gr.Markdown("""
                ---
                ### ℹ️ About SOAP Format
                
                **SOAP** is a structured method of documentation:
                
                - **S (Subjective):** Patient's description of symptoms
                - **O (Objective):** Observable/measurable findings
                - **A (Assessment):** Diagnosis or clinical impression
                - **P (Plan):** Treatment and follow-up plan
                
                ---
                
                ### 🎯 Model Information
                
                - **Base Model:** Facebook BART-Large-CNN
                - **Fine-tuned On:** Medical dialogue → SOAP note pairs
                - **Training Data:** Medical conversation dataset
                
                ---
                
                ### 🔗 API Endpoints
                
                This server also provides REST API endpoints:
                - `GET /` - API information
                - `GET /health` - Health check
                - `POST /generate` - Generate SOAP note
                - `POST /batch` - Batch generation
                
                ---
                
                ### ⚠️ Disclaimer
                
                This tool is for **demonstration and research purposes only**.
                Always consult qualified healthcare professionals for medical advice.
                Do not use for actual clinical decision-making.
                """)
        
        # Connect button to function
        generate_btn.click(
            fn=gradio_generate_soap_note,
            inputs=[
                input_dialogue,
                max_length_slider,
                min_length_slider,
                num_beams_slider,
                length_penalty_slider,
                repetition_penalty_slider
            ],
            outputs=output_soap,
            api_name="generate_soap_gradio"
        )
    
    return demo

# Create Gradio interface (but don't mount - Flask and Gradio need separate processes)
logger.info("Gradio interface created and ready")
gradio_demo = create_gradio_interface()

# Add route to redirect to Gradio (if running separately)
@app.route('/gradio')
def gradio_redirect():
    """Redirect to Gradio UI (run separately on port 7860)"""
    return jsonify({
        "message": "Gradio UI is available separately",
        "instructions": "Run 'python app.py' in another terminal to start Gradio on port 7860",
        "gradio_url": "http://localhost:7860",
        "note": "Or use the integrated Flask+Gradio server with 'python flask_api.py --with-gradio'"
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Check if --with-gradio flag is provided
    with_gradio = '--with-gradio' in sys.argv or '-g' in sys.argv
    
    port = int(os.environ.get('PORT', 5000))
    
    if with_gradio:
        # Run Flask and Gradio together using threading
        logger.info("Starting integrated Flask + Gradio server")
        logger.info(f"API endpoints available at: http://localhost:{port}/")
        logger.info(f"Gradio UI available at: http://localhost:7860/")
        
        import threading
        
        # Start Flask in a separate thread
        def run_flask():
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Start Gradio in main thread
        logger.info("Starting Gradio interface...")
        gradio_demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    else:
        # Run Flask only
        logger.info(f"Starting Flask API server on port {port}")
        logger.info(f"API endpoints available at: http://localhost:{port}/")
        logger.info(f"")
        logger.info(f"💡 Tip: To also start Gradio UI, use:")
        logger.info(f"   python flask_api.py --with-gradio")
        logger.info(f"   or run 'python app.py' in another terminal")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
