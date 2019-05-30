#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model,model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'
STRUCT_PATH = 'models/model.json'
# Load your trained model
# model = load_model(MODEL_PATH)
         # Necessary


# load json and create model
with open(STRUCT_PATH, 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(MODEL_PATH)
model._make_predict_function() 
print("Loaded model from disk")


print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = x.reshape((1,x.shape[0], x.shape[1], x.shape[2]))

    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(x)
    print(preds)
    return preds

def decode_predictions(preds, top=1):
    if preds == np.array([0]):
        return "Not Stocked"
    elif preds == np.array([2]):
        return "Stocked"
    else:
        return "Other"

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(pred_class)
        result = str(pred_class)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
