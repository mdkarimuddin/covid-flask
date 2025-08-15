# COVID-19 X-Ray Classifier (Flask)

A simple Flask web app that loads a pre-trained Keras model to classify chest X-Ray images.

## Prerequisites
- Python 3.9–3.11
- Git LFS (required to fetch the actual `model.h5` binary)

## Setup
```bash
# Clone the repo
git clone <your-repo-url>
cd <repo>

# Ensure Git LFS is installed and fetch large files
git lfs install
git lfs pull

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Running
```bash
# Option 1: Python
python app.py

# Option 2: Flask CLI
export FLASK_APP=app.py
flask run

# Option 3: Gunicorn (production-style)
gunicorn --bind 0.0.0.0:5000 app:app
```

You can control debug/host/port via environment variables:
```bash
export FLASK_DEBUG=1   # enable debug mode
export HOST=0.0.0.0
export PORT=5000
```

## Notes
- If you see an error about loading `model.h5`, ensure that Git LFS has pulled the real file (the pointer text file cannot be loaded by Keras).
- Uploads are limited to 10 MB and allowed extensions are: jpg, jpeg, png, gif.
- Model input size is set to 256x256 to match training configuration; adjust `target_size` in `app.py` if your model expects a different size.