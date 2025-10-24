# 🏥 Medical SOAP Summarization API - Modular Architecture

A production-ready Flask API and Gradio web interface for generating SOAP (Subjective, Objective, Assessment, Plan) notes from medical dialogues using a fine-tuned BART-Large-CNN model.

---

## 📁 Project Structure

```
├── app/                          # Main application package
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration and constants
│   ├── service.py               # Model loading and generation logic
│   ├── routes.py                # Flask API endpoints
│   └── gradio_ui.py             # Gradio web interface
│
├── main.py                       # Unified entry point (Flask + Gradio)
├── wsgi.py                       # Flask-only entry point (production)
├── gradio_main.py                # Gradio-only entry point
│
├── requirements.txt              # Python dependencies
├── requirements_flask.txt        # Flask-specific dependencies
├── requirements_deployment.txt   # Gradio deployment dependencies
│
├── .gitignore                    # Git ignore patterns
└── bart-large-cnn-finetuned-soap-model/  # Fine-tuned model files
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run the Application

#### **Option A: Flask API + Gradio UI (Integrated Mode)**
```bash
python main.py --with-gradio
```
- Flask API: http://localhost:5000/
- Gradio UI: http://localhost:7860/

#### **Option B: Flask API Only**
```bash
python main.py
# or
python wsgi.py
```
- Flask API: http://localhost:5000/

#### **Option C: Gradio UI Only**
```bash
python gradio_main.py
```
- Gradio UI: http://localhost:7860/

---

## 🔌 API Endpoints

### **GET /**
API information and available endpoints.

```bash
curl http://localhost:5000/
```

### **GET /health**
Health check endpoint.

```bash
curl http://localhost:5000/health
```

### **POST /generate**
Generate a SOAP note from a medical dialogue.

**Request:**
```json
{
  "dialogue": "Patient: I have severe headaches...",
  "max_length": 900,
  "min_length": 150,
  "num_beams": 4,
  "length_penalty": 1.5,
  "repetition_penalty": 1.2
}
```

**Response:**
```json
{
  "success": true,
  "soap_note": "S: Patient reports severe headaches...",
  "metadata": {
    "input_length": 450,
    "output_length": 280,
    "compression_ratio": 0.622,
    "tokens_generated": 95,
    "generation_time_seconds": 1.234,
    "device": "cuda"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Patient: I have been experiencing severe headaches for the past week. Doctor: Can you describe the pain? Patient: It is a throbbing pain on both sides of my head."}'
```

### **POST /batch**
Generate SOAP notes for multiple dialogues.

**Request:**
```json
{
  "dialogues": [
    "Dialogue 1...",
    "Dialogue 2..."
  ],
  "max_length": 900
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {"soap_note": "...", "metadata": {...}},
    {"soap_note": "...", "metadata": {...}}
  ],
  "total_processed": 2,
  "total_time_seconds": 3.456
}
```

---

## 🎨 Gradio Web Interface

Visit http://localhost:7860/ when running with `--with-gradio` flag or `gradio_main.py`.

**Features:**
- Interactive web UI for SOAP note generation
- Example medical dialogues pre-loaded
- Advanced generation parameters (adjustable sliders)
- Real-time generation statistics
- Copy-to-clipboard functionality

---

## ⚙️ Configuration

Edit `app/config.py` to customize:

```python
# Model configuration
MODEL_PATH = "./bart-large-cnn-finetuned-soap-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation parameters
GEN_MAX_LENGTH = 900
GEN_MIN_LENGTH = 150
GEN_NUM_BEAMS = 4
GEN_LENGTH_PENALTY = 1.5
GEN_REPETITION_PENALTY = 1.2

# Server configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
GRADIO_PORT = 7860
```

**Environment Variables:**
```bash
export MODEL_PATH="./bart-large-cnn-finetuned-soap-model"
export FLASK_PORT=5000
export GRADIO_PORT=7860
export LOG_LEVEL="INFO"
```

---

## 🏭 Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

# With timeout for long-running requests
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 wsgi:app
```

### Using systemd (Linux)

Create `/etc/systemd/system/soap-api.service`:

```ini
[Unit]
Description=Medical SOAP Summarization API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl start soap-api
sudo systemctl enable soap-api
sudo systemctl status soap-api
```

---

## 🐳 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000 7860

CMD ["python", "main.py", "--with-gradio"]
```

Build and run:
```bash
docker build -t soap-api .
docker run -p 5000:5000 -p 7860:7860 soap-api
```

---

## ☁️ Hugging Face Space Deployment

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Gradio" as SDK
3. Upload files:
   - `app/` directory
   - `gradio_main.py`
   - `requirements_deployment.txt` (rename to `requirements.txt`)
   - Model directory: `bart-large-cnn-finetuned-soap-model/`

4. Create `app.py` in root:
```python
from gradio_main import *
```

---

## 🧪 Testing

### Test with curl

```bash
# Health check
curl http://localhost:5000/health

# Generate SOAP note
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Patient: I have a fever. Doctor: How long have you had it?"}'
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:5000/generate",
    json={"dialogue": "Patient: I have severe headaches..."}
)

print(response.json())
```

### Use provided test script

```bash
python test_api.py
```

---

## 📦 Dependencies

**Core:**
- Flask 3.0.0+
- Gradio 4.0.0+
- Transformers 4.35.0+
- PyTorch 2.0.0+ (CUDA/CPU)
- Flask-CORS 4.0.0+

**Production:**
- Gunicorn 21.2.0+

**See `requirements.txt` for complete list.**

---

## 🔧 Module Details

### `app/config.py`
Central configuration for all settings, paths, and constants.

### `app/service.py`
- Singleton model loading pattern
- `generate_soap_note()`: Core generation logic
- `get_model_info()`: Model status information

### `app/routes.py`
- Flask Blueprint with all API endpoints
- Request validation and error handling
- Batch processing support

### `app/gradio_ui.py`
- Gradio Blocks interface
- Example dialogues
- Advanced parameter controls

### `app/__init__.py`
- Flask application factory
- Blueprint registration
- CORS configuration

---

## 🔄 Migration from `flask_api.py`

If you have the old monolithic `flask_api.py`:

**Old usage:**
```bash
python flask_api.py --with-gradio
```

**New usage (identical functionality):**
```bash
python main.py --with-gradio
```

**Benefits of modular structure:**
- ✅ Better code organization and maintainability
- ✅ Easier testing and debugging
- ✅ Separation of concerns (API, UI, logic)
- ✅ More flexible deployment options
- ✅ Follows Flask best practices

---

## ⚠️ Important Notes

1. **Model Loading**: Model is lazy-loaded on first request to avoid issues with multi-worker setups.

2. **Threading Mode**: When using `--with-gradio`, Flask runs in a background thread while Gradio runs in the main thread.

3. **Port Configuration**: Flask (5000) and Gradio (7860) use separate ports in integrated mode.

4. **CUDA Support**: Automatically detects GPU availability. Set `CUDA_VISIBLE_DEVICES` to control GPU usage.

5. **Batch Size Limit**: Maximum 100 dialogues per batch request (configurable in `config.py`).

---

## 🆘 Troubleshooting

### "Model not found" error
```bash
# Verify model path
ls -la bart-large-cnn-finetuned-soap-model/

# Set correct path
export MODEL_PATH="./bart-large-cnn-finetuned-soap-model"
```

### Port already in use
```bash
# Change ports in config.py or via environment variables
export FLASK_PORT=5001
export GRADIO_PORT=7861
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CUDA out of memory
```bash
# Use CPU mode
export CUDA_VISIBLE_DEVICES=""
```

---

## 📄 License

This project is for educational and research purposes.

---

## 👨‍⚕️ Disclaimer

**This tool is for demonstration and research purposes only.**

- ⚠️ Not intended for actual clinical use
- ⚠️ Do not use for medical decision-making
- ⚠️ Always consult qualified healthcare professionals
- ⚠️ Generated SOAP notes should be reviewed by medical professionals

---

## 📧 Support

For issues, questions, or contributions, please refer to the project repository.

---

## 🎯 Summary of Entry Points

| File | Purpose | Command | Ports |
|------|---------|---------|-------|
| `main.py` | Unified server | `python main.py --with-gradio` | 5000, 7860 |
| `wsgi.py` | Flask API only | `python wsgi.py` | 5000 |
| `gradio_main.py` | Gradio UI only | `python gradio_main.py` | 7860 |

Choose the entry point based on your deployment needs!
