from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import get_custom_objects
from efficientnet.keras import EfficientNetB0, preprocess_input
import numpy as np
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define the target size for resizing the input image
target_size = (256, 256)

# Define custom objects for EfficientNetB0
custom_objects = {
    'swish': get_custom_objects()['swish'],
    'FixedDropout': get_custom_objects()['FixedDropout']
}
get_custom_objects().update(custom_objects)

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
    app.run(debug=True)

