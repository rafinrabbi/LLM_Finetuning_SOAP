# 🚀 Quick Start Guide - Flask API + Gradio Server

## ▶️ Start Options

### Option 1: Flask API Only
```bash
python flask_api.py
```
- API: http://localhost:5000/

### Option 2: Flask API + Gradio UI (Integrated)
```bash
python flask_api.py --with-gradio
```
- API: http://localhost:5000/
- Gradio: http://localhost:7860/

### Option 3: Using Startup Script
```bash
./start_server.sh              # Flask only
./start_server.sh --with-gradio  # Flask + Gradio
```

---

## 🌐 Available Services

### 🔌 Flask REST API (Port 5000)
- `GET /` - API info
- `GET /health` - Health check
- `POST /generate` - Generate SOAP note
- `POST /batch` - Batch generation

### 🎨 Gradio Web UI (Port 7860)
- Interactive interface
- Example dialogues
- Parameter controls
- Real-time generation

---

## 🧪 Quick Test

**Test API:**
```bash
curl http://localhost:5000/health
```

**Test Gradio:**
Open http://localhost:7860/ in browser

---

## 💡 Which Mode to Use?

| Mode | Command | Best For |
|------|---------|----------|
| Flask only | `python flask_api.py` | Production API |
| Both | `python flask_api.py --with-gradio` | Development/Demos |
| Separate | `python app.py` (other terminal) | Debugging |

---

**Ready? Start with:**
```bash
python flask_api.py --with-gradio
```
