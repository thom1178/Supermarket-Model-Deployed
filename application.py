from __future__ import division, print_function
# coding=utf-8
import sys
import os
# import glob
# import re
import numpy as np

sys.path.append('/usr/local/lib64/python3.6/site-packages')
import cv2
import localize
# Keras
# from keras.applications.vgg16 import preprocess_input
from keras.models import load_model,model_from_json
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
application = app = Flask(__name__)

w = 400
h = 400
epsilon = .1
threshold = .5

# Model saved with Keras model.save()
MODEL_PATH = 'models/classification.h5'
STRUCT_PATH = 'models/classification.json'
MODEL_PATH_LOCAL = 'models/localization.h5'
STRUCT_PATH_LOCAL = 'models/localization.json'
# Load your trained model
# model = load_model(MODEL_PATH)
         # Necessary

hists = os.listdir('static/plots')
hists = ['plots/' + file for file in hists if not file.startswith(".") and file != "tmp"]

## Clean Prod Directory
filelist = [ f for f in os.listdir("static/plots/tmp/") if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") ]
for f in filelist:
    os.remove(os.path.join("static/plots/tmp/", f))


# load json and create model
with open(STRUCT_PATH, 'r') as json_file:
    loaded_model_json = json_file.read()


model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(MODEL_PATH)

new_model = localize.to_fully_conv(model)


model._make_predict_function() 
new_model._make_predict_function() 
print("Loaded model from disk")


print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    #x = image.img_to_array(img)
    #x = x.reshape((1,x.shape[0], x.shape[1], x.shape[2]))

    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)
    x = localize.process_pred_img(img, w = w, h= h )
    print("Predicting...")
    preds = model.predict(x)
    print(preds)
    print("Localizing...")
    localized = localize.localizee(model=new_model, 
                     unscaled = img,
                     W = w, 
                     H = h,
                     EPSILON = epsilon,
                     THRESHOLD = threshold)
    print("Supressing...")
    suppressed_boxes = localize.non_max_suppression_fast(boxes = localized, overlapThresh = .02)
    for box in suppressed_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 20)
    img_path_new = img_path.split("/")[-1]
    print("static/plots/tmp/localized_" + img_path_new)
    
    cv2.imwrite("static/plots/tmp/localized_" + img_path_new, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return {"pred": preds , "path": img_path_new}

def decode_predictions(preds, top=1):
    if preds == np.array([0]):
        return "Not-Stocked"
    elif preds == np.array([2]):
        return "Stocked"
    else:
        return "Other"

@app.route('/', methods=['GET'])
def index():
    # Main page
    print(hists)
    return render_template('index.html', hists = enumerate(hists))


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Try File from post request
        try:
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds_file = model_predict(file_path, model)
            preds = preds_file["pred"]
            file_path_localized = preds_file["path"]
            
            # Process your result for human
            pred_class = preds.argmax(axis=-1)            # Simple argmax
            pred_class = decode_predictions(pred_class)
            result = str(pred_class)               # Convert to string
            os.remove(file_path)
            print(result + " " + file_path_localized)
            return result + " " + file_path_localized
        except: # If the first fails, must be a selected photo
            selected = int(request.form.get('selectedPhoto'))
            preds_file = model_predict("static/" + hists[selected], model)
            preds = preds_file["pred"]
            file_path_localized = preds_file["path"]
            
            # Process your result for human
            pred_class = preds.argmax(axis=-1)            # Simple argmax
            pred_class = decode_predictions(pred_class)
            result = str(pred_class)               # Convert to string
            print(result + " " + file_path_localized)
            return result + " " + file_path_localized
            
    return None


@app.route('/get_image')
def get_image():
    path = request.args.get('p')
    filename = "static/plots/tmp/localized_" + path
    print(filename)
    exists = os.path.isfile(filename)
    if exists:
        return send_file(filename, mimetype='image/gif')
    return send_file("error.gif", mimetype='image/gif')

if __name__ == '__main__':
    
    #app.run()

    # Serve the app with gevent
    http_server = WSGIServer((''), app)
    http_server.serve_forever()
