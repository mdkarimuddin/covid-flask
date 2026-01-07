from flask import Flask, request, render_template
import os
import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Monkey-patch keras backend to fix 'sigmoid' attribute error in efficientnet
import keras
keras.backend.sigmoid = tf.keras.activations.sigmoid

# Try to import efficientnet preprocess_input, with fallback
try:
    from efficientnet.keras import preprocess_input
except ImportError:
    try:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except ImportError:
        # Fallback preprocessing function
        def preprocess_input(x):
            return (x / 255.0 - 0.5) * 2.0

# Initialize the Flask app
app = Flask(__name__)

# Define the target size for resizing the input image
target_size = (256, 256)

# Load the trained model (fail with a helpful message if missing)
MODEL_PATH = 'model_fixed.keras'
if not os.path.exists(MODEL_PATH):
    sys.stderr.write(f"ERROR: Repaired model '{MODEL_PATH}' not found. Run repair_model.py to generate it.\n")
    # don't crash during import; set model to None and handle at request time
    model = None
else:
    try:
        # Provide modern path to 'sigmoid' to fix runtime error
        custom_objects = {'sigmoid': tf.keras.activations.sigmoid}
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        sys.stderr.write(f"ERROR loading {MODEL_PATH}: {e}\n")
        model = None

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the user
    file = request.files['file']

    # Load the image and resize it to the target size
    if model is None:
        return render_template('result.html', prediction='Model not loaded. See server logs for details')

    img = load_img(io.BytesIO(file.read()), target_size=target_size)
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)

    # Make the prediction using the loaded model
    prediction = model.predict(np.expand_dims(img_arr, axis=0))[0][0]

    # Determine the class of the prediction
    if prediction < 0.5:
        class_name = 'Negative'
    else:
        class_name = 'Positive'

    # Render the prediction result on a new page
    return render_template('result.html', prediction=class_name)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)

