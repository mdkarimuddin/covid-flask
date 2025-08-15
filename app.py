from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import get_custom_objects
from efficientnet.keras import preprocess_input
import efficientnet.keras as efn
import numpy as np
import io
import os

# Initialize the Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

# Register EfficientNet custom objects before loading the model
get_custom_objects().update(efn.CUSTOM_OBJECTS)

# Try to load the trained model (allow app to start even if unavailable)
model = None
model_load_error = None
try:
    model = load_model('model.h5')
except Exception:
    model_load_error = 'Model not loaded. Ensure Git LFS pulled the real model.h5 (run: git lfs install && git lfs pull).'

# Define the target size for resizing the input image
target_size = (256, 256)

# Upload constraints
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html', error=model_load_error)

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error=model_load_error or 'Model is not available.')

    # Validate the uploaded file
    if 'file' not in request.files:
        return render_template('index.html', error='No file part in the request.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', error='Unsupported file type. Please upload a JPG or PNG image.')

    try:
        # Load the image and resize it to the target size (force RGB)
        img = load_img(io.BytesIO(file.read()), target_size=target_size, color_mode='rgb')
        img_arr = img_to_array(img)
        img_arr = preprocess_input(img_arr)

        # Make the prediction using the loaded model
        prediction = model.predict(np.expand_dims(img_arr, axis=0))[0][0]
    except Exception:
        return render_template('index.html', error='Failed to process the image. Please try another file.')

    # Determine the class of the prediction
    class_name = 'Negative' if prediction < 0.5 else 'Positive'

    # Render the prediction result on a new page
    return render_template('result.html', prediction=class_name)

# Run the Flask app
if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5000'))
    app.run(host=host, port=port, debug=debug)

