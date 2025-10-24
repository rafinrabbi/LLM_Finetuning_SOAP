# 🔄 Migration Guide: flask_api.py → Modular Structure

This guide helps you transition from the monolithic `flask_api.py` to the new modular architecture.

---

## 📋 Quick Migration Checklist

- [ ] Backup your old `flask_api.py`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test new structure: `python main.py --with-gradio`
- [ ] Verify API endpoints work: `curl http://localhost:5000/health`
- [ ] Verify Gradio UI works: Visit http://localhost:7860/
- [ ] Update deployment scripts/documentation
- [ ] (Optional) Remove old `flask_api.py`

---

## 🆚 What Changed?

### **Before: Monolithic Structure**
```
flask_api.py (679 lines)
├── Imports and configuration
├── Model loading
├── Generation logic
├── Flask routes
├── Gradio interface
└── Threading logic
```

### **After: Modular Structure**
```
app/
├── __init__.py        # Flask app factory
├── config.py          # All configuration
├── service.py         # Model & generation logic
├── routes.py          # API endpoints
└── gradio_ui.py       # Gradio interface

main.py               # Unified entry point
wsgi.py               # Flask-only entry point
gradio_main.py        # Gradio-only entry point
```

---

## 🔀 Command Equivalents

| Old Command | New Command | Notes |
|-------------|-------------|-------|
| `python flask_api.py` | `python main.py` | Flask API only |
| `python flask_api.py --with-gradio` | `python main.py --with-gradio` | Flask + Gradio |
| `gunicorn flask_api:app` | `gunicorn wsgi:app` | Production mode |
| N/A | `python gradio_main.py` | Gradio only (new) |

---

## 📍 Where Did Everything Go?

### **Configuration Variables**
**Old:** Top of `flask_api.py` (lines 1-30)
```python
MODEL_PATH = os.environ.get("MODEL_PATH", "...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**New:** `app/config.py`
```python
MODEL_PATH = os.environ.get("MODEL_PATH", "...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_MAX_LENGTH = 900
GEN_MIN_LENGTH = 150
# ... all configuration in one place
```

---

### **Model Loading**
**Old:** `flask_api.py` (lines 30-60)
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
```

**New:** `app/service.py`
```python
def load_model_once():
    global _tokenizer, _model, _generation_config
    # Singleton pattern with lazy loading
```

---

### **Generation Function**
**Old:** `flask_api.py` - `generate_soap_note()`
```python
def generate_soap_note(dialogue, max_length=900, ...):
    # Generation logic
```

**New:** `app/service.py` - `generate_soap_note()`
```python
def generate_soap_note(dialogue, max_length=900, ...):
    tokenizer, model, gen_config = load_model_once()
    # Same logic, now modular
```

---

### **Flask Routes**
**Old:** `flask_api.py` (lines 100-400)
```python
@app.route('/')
def home():
    return jsonify({...})

@app.route('/generate', methods=['POST'])
def generate():
    # Logic
```

**New:** `app/routes.py`
```python
api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def home():
    return jsonify({...})

@api_bp.route('/generate', methods=['POST'])
def generate():
    # Same logic
```

---

### **Gradio Interface**
**Old:** `flask_api.py` - `create_gradio_interface()`
```python
def create_gradio_interface():
    with gr.Blocks() as demo:
        # UI components
    return demo
```

**New:** `app/gradio_ui.py`
```python
def create_gradio_interface():
    with gr.Blocks() as demo:
        # Same UI components
    return demo

def launch_gradio(server_name, server_port, share):
    # Launch logic
```

---

### **Threading Logic**
**Old:** `flask_api.py` (bottom of file)
```python
if __name__ == '__main__':
    if '--with-gradio' in sys.argv:
        # Threading code
```

**New:** `main.py`
```python
if __name__ == '__main__':
    with_gradio = '--with-gradio' in sys.argv
    
    if with_gradio:
        # Same threading logic
```

---

## 🔧 Configuration Changes

### **Environment Variables**
All environment variables work exactly the same:

```bash
# Still works the same way
export MODEL_PATH="./bart-large-cnn-finetuned-soap-model"
export FLASK_PORT=5000
export GRADIO_PORT=7860
export LOG_LEVEL="INFO"
```

### **New Configuration File**
Now you can also edit `app/config.py` directly:

```python
# app/config.py
MODEL_PATH = "./my-custom-model-path"  # No need for env vars
FLASK_PORT = 8080
GRADIO_PORT = 7861
```

---

## 🚀 Running the Application

### **Development Mode**

```bash
# Flask API + Gradio UI (RECOMMENDED)
python main.py --with-gradio
# Flask: http://localhost:5000/
# Gradio: http://localhost:7860/

# Flask API only
python main.py
# or
python wsgi.py

# Gradio UI only (NEW)
python gradio_main.py
```

### **Production Mode**

```bash
# Gunicorn (Flask API only)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 wsgi:app

# With Gradio in separate process
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app &
python gradio_main.py &
```

---

## 🧪 Testing Migration

### **1. Health Check**
```bash
curl http://localhost:5000/health
```

Expected output:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": 1234567890.123
}
```

### **2. Generate Endpoint**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Patient: I have a headache. Doctor: How long?"}'
```

### **3. Gradio UI**
Visit http://localhost:7860/ and test the interface.

---

## 📦 Dependencies

No changes to dependencies! Use the same `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or use specific requirement files:
- `requirements_flask.txt` - Flask API only
- `requirements_deployment.txt` - Gradio deployment

---

## 🐛 Common Issues

### **Issue: "Module 'app' not found"**
**Solution:** Make sure you're in the project root directory:
```bash
cd /home/RA001/Documents/Assesment\ FInal\ LLM\ finetuning/
python main.py --with-gradio
```

### **Issue: "Import errors"**
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### **Issue: "Model not loading"**
**Solution:** Check model path:
```bash
ls -la bart-large-cnn-finetuned-soap-model/
export MODEL_PATH="./bart-large-cnn-finetuned-soap-model"
```

### **Issue: "Port already in use"**
**Solution:** Change ports in `app/config.py` or use environment variables:
```bash
export FLASK_PORT=5001
export GRADIO_PORT=7861
```

---

## ✅ Benefits of Modular Structure

### **1. Better Organization**
- Each module has a single responsibility
- Easier to find and modify specific functionality

### **2. Easier Testing**
```python
# Test model service independently
from app.service import generate_soap_note
result = generate_soap_note("test dialogue")

# Test routes independently
from app import create_app
app = create_app()
with app.test_client() as client:
    response = client.get('/health')
```

### **3. Flexible Deployment**
- Run Flask and Gradio separately or together
- Scale Flask workers without affecting Gradio
- Deploy Gradio to Hugging Face Space easily

### **4. Maintainability**
- Clear separation of concerns
- Easy to add new features
- Follows Flask best practices

### **5. Reusability**
```python
# Use service in other projects
from app.service import generate_soap_note

# Use Gradio UI standalone
from app.gradio_ui import create_gradio_interface
```

---

## 🔄 Rollback Plan

If you need to go back to `flask_api.py`:

1. Keep your backup of `flask_api.py`
2. Run: `python flask_api.py --with-gradio`
3. Everything works as before!

The old file is **fully functional** - you can use either structure.

---

## 📈 Next Steps

1. **Familiarize yourself** with the new structure
2. **Test thoroughly** in your environment
3. **Update deployment scripts** to use new entry points
4. **Update documentation** for your team
5. **Consider removing** `flask_api.py` after successful migration

---

## 💡 Tips

- **Gradual Migration:** You can run both old and new versions side-by-side on different ports
- **Version Control:** Commit the modular structure before removing old files
- **Documentation:** Update any internal docs or wikis
- **Team Training:** Share this guide with your team

---

## 📞 Need Help?

If you encounter issues during migration:

1. Check this guide's "Common Issues" section
2. Review `README_MODULAR.md` for detailed documentation
3. Verify all files are in the correct locations
4. Test each component independently

---

## 🎯 Summary

| Aspect | Old | New |
|--------|-----|-----|
| **Structure** | Monolithic | Modular |
| **Lines of Code** | 679 lines in 1 file | ~150 lines per module |
| **Maintainability** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Testability** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Deployment Flexibility** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Functionality** | ✅ Full | ✅ Full (identical) |

**The functionality is 100% identical - just better organized!**

---

## ✨ Conclusion

The modular structure provides the **same functionality** with **better organization**. All your existing commands, API endpoints, and features work exactly the same way.

**Happy coding! 🚀**
