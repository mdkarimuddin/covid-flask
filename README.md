# COVID-19 X-ray Classification Flask App

Small Flask web application to serve a chest X-ray classifier. Upload an X-ray image and get a COVID-19 Positive/Negative prediction.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Deep%20Learning%20%7C%20CNN-orange.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)

Why this repo won't run out-of-the-box
- The trained model weights (`model.h5`) are large and are tracked via Git LFS in the original repo; the clone here contains only a small LFS pointer file. You need to fetch the real model file before the app can make predictions.

Quick start (recommended)

1) Create a Python virtual environment and install dependencies

```bash
cd covid-flask
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2) Get the model weights (pick one)

- If you have `git-lfs` installed locally

```bash
# install git-lfs (macOS: `brew install git-lfs`, Ubuntu: `sudo apt install git-lfs`)
git lfs install
git lfs pull
```

- If you don't want to use Git LFS, download `model.h5` from a Release or cloud storage and place it in the repo root:

```bash
curl -L -o model.h5 "https://example.com/path/to/model.h5"
```

3) Run the app

```bash
python app.py
# then open http://127.0.0.1:5000
```

4) Troubleshooting
- If import errors occur when starting the server, run `python check_env.py` to see which packages are missing.
- If you get "Unknown layer" errors when loading the model, the `app.py` now attempts to merge EfficientNet custom objects to reduce that risk; if you still see errors, the model was likely saved with custom layers not present in `efficientnet` and you'll need to provide custom_objects at load time.

Hosting the model
- Best practice: upload the large `model.h5` to a GitHub Release, S3, or other cloud storage and link to it in this README. That way users can download the weights without using LFS.

## 👤 Author

**Md Karim Uddin, PhD**  
PhD Veterinary Medicine | MEng Big Data Analytics  
Postdoctoral Researcher, University of Helsinki

- GitHub: [@mdkarimuddin](https://github.com/mdkarimuddin)
- LinkedIn: [Md Karim Uddin, PhD](https://www.linkedin.com/in/md-karim-uddin-phd-aa87649a/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repo if you found it useful!**

*Built to demonstrate Flask deployment of deep learning models for medical image classification.*
