# app/gradio_ui.py
"""
Gradio UI module for Medical SOAP Summarization
================================================
Creates and manages the Gradio web interface.
"""

import gradio as gr
from .service import generate_soap_note
from .config import DEVICE
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE DIALOGUES
# ============================================================================

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

# ============================================================================
# GRADIO WRAPPER FUNCTION
# ============================================================================

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
        logger.exception("Error in Gradio generation")
        return f"❌ Error generating summary: {str(e)}"

# ============================================================================
# CREATE GRADIO INTERFACE
# ============================================================================

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

# ============================================================================
# LAUNCH GRADIO
# ============================================================================

def launch_gradio(server_name="0.0.0.0", server_port=7860, share=False):
    """Launch the Gradio interface"""
    logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True
    )
    
    return demo
