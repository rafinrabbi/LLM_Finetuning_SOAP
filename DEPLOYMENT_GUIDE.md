# 🚀 Deployment Guide - Medical SOAP Summarization Model

This guide provides step-by-step instructions for deploying the fine-tuned BART model for medical SOAP note generation.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Option 1: Hugging Face Space (Recommended)](#option-1-hugging-face-space-recommended)
4. [Option 2: Flask API](#option-2-flask-api)
5. [Option 3: Local Deployment](#option-3-local-deployment)
6. [API Usage Examples](#api-usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ Fine-tuned model saved in `bart-large-cnn-finetuned-soap-model/` directory
- ✅ All required files:
  - `model.safetensors` or `pytorch_model.bin`
  - `config.json`
  - `tokenizer.json`
  - `vocab.json`
  - `generation_config.json`
- ✅ Hugging Face account (for HF Space deployment)
- ✅ Git installed on your machine

---

## Deployment Options

### Comparison Table

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hugging Face Space** | Free, Easy, Public URL, No server management | Limited resources | Quick demos, sharing |
| **Flask API** | Full control, Custom logic | Need server, Maintenance | Production APIs |
| **Local** | Full control, Privacy | No public access | Development, testing |

---

## Option 1: Hugging Face Space (Recommended)

Hugging Face Spaces provides free hosting with a beautiful Gradio interface.

### Step 1: Prepare Your Files

Ensure you have these files in your project directory:

```
your-project/
├── app.py                                    # Gradio interface
├── requirements_deployment.txt               # Dependencies
└── bart-large-cnn-finetuned-soap-model/     # Your model
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── vocab.json
    ├── merges.txt
    ├── special_tokens_map.json
    └── tokenizer_config.json
```

### Step 2: Create Hugging Face Space

1. **Go to Hugging Face**: https://huggingface.co/spaces

2. **Click "Create new Space"**

3. **Fill in details:**
   - **Space name**: `medical-soap-summarization`
   - **License**: Choose appropriate license
   - **Select SDK**: **Gradio**
   - **Space hardware**: CPU Basic (free) or GPU (paid)

4. **Click "Create Space"**

### Step 3: Upload Files via Git

#### Option A: Using Git Command Line

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR-USERNAME/medical-soap-summarization
cd medical-soap-summarization

# Copy your files
cp /path/to/your/app.py .
cp /path/to/your/requirements_deployment.txt ./requirements.txt

# Copy model directory
cp -r /path/to/bart-large-cnn-finetuned-soap-model .

# Commit and push
git add .
git commit -m "Initial deployment of medical SOAP summarization model"
git push
```

#### Option B: Using Web Interface

1. In your Space, click **"Files and versions"**
2. Click **"Add file"** → **"Upload files"**
3. Upload:
   - `app.py`
   - `requirements.txt` (rename from `requirements_deployment.txt`)
   - All files from `bart-large-cnn-finetuned-soap-model/` directory

### Step 4: Configure Space (Optional)

Create a `README.md` in your Space:

```markdown
---
title: Medical SOAP Note Generator
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Medical SOAP Note Generator

Generate structured SOAP notes from medical dialogues using fine-tuned BART-Large-CNN.

## Usage

1. Enter a medical dialogue in the text area
2. Adjust generation parameters (optional)
3. Click "Generate SOAP Note"
4. View and copy the generated SOAP note

## Model

- **Base**: facebook/bart-large-cnn
- **Fine-tuned on**: Medical dialogue → SOAP note pairs
- **Performance**: ROUGE-L: 0.XX | BLEU: 0.XX

## Disclaimer

This tool is for demonstration purposes only. Not for clinical use.
```

### Step 5: Wait for Build

- Hugging Face will automatically build your Space
- Check the **"Logs"** tab for any errors
- Once built, your Space will be live at: `https://huggingface.co/spaces/YOUR-USERNAME/medical-soap-summarization`

### Step 6: Test Your Deployment

1. Visit your Space URL
2. Try the example dialogues
3. Test custom inputs
4. Share the link!

---

## Option 2: Flask API

Deploy a REST API for programmatic access.

### Step 1: Prepare Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_flask.txt
```

### Step 2: Test Locally

```bash
# Run Flask app
python flask_api.py
```

The API will be available at `http://localhost:5000`

### Step 3: Test API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Generate SOAP note
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Patient: I have a headache. Doctor: How long have you had it?"
  }'
```

### Step 4: Deploy to Cloud

#### Option A: Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn flask_api:app" > Procfile

# Create runtime.txt
echo "python-3.10.12" > runtime.txt

# Deploy
heroku create your-app-name
git add .
git commit -m "Deploy Flask API"
git push heroku main
```

#### Option B: Google Cloud Run

```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.10-slim
WORKDIR /app
COPY requirements_flask.txt .
RUN pip install -r requirements_flask.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:8080", "flask_api:app"]
EOF

# Deploy
gcloud run deploy medical-soap-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Option C: AWS Lambda (Serverless)

Use AWS SAM or Serverless Framework with API Gateway.

---

## Option 3: Local Deployment

For development and testing.

### Step 1: Run Gradio App Locally

```bash
# Install dependencies
pip install -r requirements_deployment.txt

# Run app
python app.py
```

Access at: `http://localhost:7860`

### Step 2: Share Temporarily

```python
# In app.py, change launch parameters:
demo.launch(share=True)  # Creates a public temporary link
```

---

## API Usage Examples

### Python Client

```python
import requests

# API endpoint
url = "http://your-api-url.com/generate"

# Request payload
data = {
    "dialogue": """
        Patient: I've been having chest pain for 2 days.
        Doctor: Can you describe the pain?
        Patient: It's a sharp pain on the left side.
    """,
    "max_length": 900,
    "min_length": 150
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print("SOAP Note:")
print(result['soap_note'])
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

const data = {
    dialogue: `
        Patient: I've been having chest pain for 2 days.
        Doctor: Can you describe the pain?
        Patient: It's a sharp pain on the left side.
    `,
    max_length: 900,
    min_length: 150
};

axios.post('http://your-api-url.com/generate', data)
    .then(response => {
        console.log('SOAP Note:', response.data.soap_note);
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

### cURL

```bash
curl -X POST http://your-api-url.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Patient: I have a fever. Doctor: How high is your temperature?",
    "max_length": 900,
    "min_length": 150,
    "num_beams": 4
  }'
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found Error

**Problem**: `OSError: model not found`

**Solution**:
- Ensure model directory is correctly uploaded
- Check file paths in `app.py` or `flask_api.py`
- Verify all model files are present

#### 2. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size (for Flask API)
- Use CPU instead of GPU
- Use smaller beam size
- Enable gradient checkpointing

#### 3. Gradio Interface Not Loading

**Problem**: Space stuck on "Building"

**Solution**:
- Check logs for errors
- Verify `requirements.txt` is correct
- Ensure `app.py` has no syntax errors
- Try with smaller model first

#### 4. Slow Generation

**Problem**: Takes too long to generate

**Solution**:
- Reduce `max_length`
- Reduce `num_beams`
- Use GPU hardware (paid tier)
- Optimize tokenization

#### 5. API Returns 500 Error

**Problem**: Internal server error

**Solution**:
- Check server logs
- Validate input format
- Ensure model is properly loaded
- Check parameter ranges

---

## Performance Optimization

### For Hugging Face Space

```python
# In app.py
# Enable caching
@functools.lru_cache(maxsize=100)
def generate_soap_note(dialogue):
    # ... generation code

# Use queue
demo.queue(concurrency_count=3)
```

### For Flask API

```python
# Use batch processing
# Implement caching with Redis
# Use async processing with Celery
```

---

## Security Considerations

1. **Input Validation**: Always validate and sanitize user inputs
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Authentication**: Add API keys for production
4. **HTTPS**: Use HTTPS for encrypted communication
5. **Monitoring**: Log requests and monitor usage

---

## Next Steps

1. ✅ Deploy to Hugging Face Space
2. ✅ Test with various inputs
3. ✅ Share with stakeholders
4. ✅ Gather feedback
5. ✅ Iterate and improve

---

## Support

For issues or questions:
- Check Hugging Face Spaces documentation: https://huggingface.co/docs/hub/spaces
- Review Flask documentation: https://flask.palletsprojects.com/
- Contact: [Your Email]

---

**Last Updated**: October 2025
