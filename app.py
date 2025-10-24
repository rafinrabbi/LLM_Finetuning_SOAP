"""
Medical SOAP Summarization - Gradio Interface for Hugging Face Space
=====================================================================

This Gradio app provides an interactive interface for generating SOAP notes
from medical dialogues using a fine-tuned BART-Large-CNN model.

Author: [Your Name]
Contact: [Your Email]
Date: October 2025
"""

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import os
from pathlib import Path

# ============================================================================
# MODEL LOADING
# ============================================================================

print("🚀 Loading fine-tuned BART model...")

# Define model path
MODEL_PATH = "./bart-large-cnn-finetuned-soap-model"

# Check if model exists locally, otherwise provide instructions
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("📦 Please ensure the model is uploaded to your Hugging Face Space")
    print("   OR use the model from Hugging Face Hub: 'your-username/model-name'")
    # Fallback to base model for demonstration
    MODEL_PATH = "facebook/bart-large-cnn"
    print(f"⚠️  Using base model for demonstration: {MODEL_PATH}")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    
    # Load generation config if available
    try:
        generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
        print("✅ Generation config loaded")
    except:
        generation_config = None
        print("⚠️  Using default generation config")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_soap_note(dialogue: str, 
                       max_length: int = 900, 
                       min_length: int = 150,
                       num_beams: int = 4,
                       length_penalty: float = 1.5,
                       repetition_penalty: float = 1.2) -> str:
    """
    Generate SOAP note from medical dialogue.
    
    Args:
        dialogue: Medical dialogue text
        max_length: Maximum length of generated summary
        min_length: Minimum length of generated summary
        num_beams: Number of beams for beam search
        length_penalty: Length penalty for generation (0.5-2.0)
        repetition_penalty: Repetition penalty (1.0-2.0)
    
    Returns:
        Generated SOAP note
    """
    
    if not dialogue or len(dialogue.strip()) == 0:
        return "⚠️ Please enter a medical dialogue to summarize."
    
    try:
        # Prepare input
        prompt = "Summarize the following medical dialogue into a concise SOAP note:\n" + dialogue
        inputs = tokenizer(prompt, 
                          return_tensors="pt", 
                          max_length=1024, 
                          truncation=True).to(device)
        
        # Generate
        with torch.no_grad():
            if generation_config:
                generation_config.decoder_start_token_id = model.config.decoder_start_token_id
                # Use saved generation config with overrides
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
                # Use manual parameters
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
        
        # Add statistics
        stats = f"\n\n📊 **Statistics:**\n"
        stats += f"- Input length: {len(dialogue)} characters\n"
        stats += f"- Output length: {len(summary)} characters\n"
        stats += f"- Compression ratio: {(len(summary)/len(dialogue)*100):.1f}%\n"
        stats += f"- Tokens generated: {len(outputs[0])} tokens"
        
        return summary + stats
        
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"

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
# GRADIO INTERFACE
# ============================================================================

# Create Gradio interface
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
            - **Performance:** ROUGE-L: ~0.XX | BLEU: ~0.XX
            
            ---
            
            ### ⚠️ Disclaimer
            
            This tool is for **demonstration and research purposes only**.
            Always consult qualified healthcare professionals for medical advice.
            Do not use for actual clinical decision-making.
            """)
    
    # Connect button to function
    generate_btn.click(
        fn=generate_soap_note,
        inputs=[
            input_dialogue,
            max_length_slider,
            min_length_slider,
            num_beams_slider,
            length_penalty_slider,
            repetition_penalty_slider
        ],
        outputs=output_soap,
        api_name="generate_soap"
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Hugging Face Space port
        share=False,             # Don't create public link (HF Space handles this)
        show_error=True
    )
