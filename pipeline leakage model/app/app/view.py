from app import app 
from flask import render_template, request, jsonify, session
import tensorflow as tf
import numpy as np
import os 



# connecting to model 
app.config['MODEL_PATH'] =  os.path.join('app', 'static', 'model', 'pipeline_model.h5')
app.config['SECRET_KEY'] = 'siodshosidp'


@app.route('/')
def index(): 
    
    model = load_prediction_model(app.config['MODEL_PATH'])
    print(model.summary())

    return render_template('index.html')


@app.route('/get-pipe-leakage', methods=['POST'])
def pipe_leakage_handler(): 
    data = request.json
    extract_data = data['data']
    extract_data = [float(xt) for xt in extract_data]
    model = load_prediction_model(app.config['MODEL_PATH'])
    # make prediction 
    prediction = detect_pipline_leakage(extract_data, extract_data, model)
    mapper = prediction_mappping(prediction)
    print(mapper)
    # returned_data = dict(zip(prediction, mapper))
    returned_data = [int(prediction[0]), int((mapper[0])), int(prediction[1]) ,float(mapper[1])]
    print(returned_data)
    return jsonify({'data':returned_data})





# the AI model functions 
# =============== MODEL HELPER SECTION ======================
def load_prediction_model(model_path):
    lm = tf.keras.models.load_model(model_path)
    lm.compile(optimizer='adam',
              loss={'loc_output': 'sparse_categorical_crossentropy', 'size_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])
    return lm


def detect_pipline_leakage(loc_f, size_f, model):

    manual_input_loc = np.array([loc_f])  # Example input for location
    manual_input_size = np.array([size_f])  # Example input for size
    # Make predictions
    predictions_loc, predictions_size = model.predict([manual_input_loc, manual_input_size])

    return np.argmax(predictions_loc), np.argmax(predictions_size)


# prediction = detect_pipline_leakage(sample, sample, pip_model)
# print(prediction)

def prediction_mappping(prediction):
    # loc_mapper= {0: '15 [m]', 1: '30 [m]', 2:'50 [m]'}
    # size_mapper={0:'0.12 [l/s]', 1:'0.31 [l/s]', 2:'0.65 [l/s]', 3:'0.29 [l/s]', 4:'0.39 [l/s]', 5:'0.60 [l/s]', 6:'0.09 [l/s]', 7:'0.37 [l/s]', 8:'0.60 [l/s]'}
    loc_mapper= {0: '15', 1: '30', 2:'50'}
    size_mapper={0:'0.12', 1:'0.31', 2:'0.65', 3:'0.29', 4:'0.39', 5:'0.60', 6:'0.09', 7:'0.37', 8:'0.60'}

    loc, size = loc_mapper[prediction[0]], size_mapper[prediction[1]]

    return loc, size

# prediction_mappping(prediction)