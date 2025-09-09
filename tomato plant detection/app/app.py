import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os 
import cv2 as cv

# Load a pre-trained model (or your custom model)
# Replace with your model path or load a sample model like MobileNet for demonstration
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Title of the app
st.title("Tomato Dieases Classification App")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

label_converter = {2: 'Tomato___Late_blight',
 9: 'Tomato___healthy',
 1: 'Tomato___Early_blight',
 4: 'Tomato___Septoria_leaf_spot',
 7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 0: 'Tomato___Bacterial_spot',
 6: 'Tomato___Target_Spot',
 8: 'Tomato___Tomato_mosaic_virus',
 3: 'Tomato___Leaf_Mold',
 5: 'Tomato___Spider_mites Two-spotted_spider_mite'}



# load and compile model 
def load_prediction_model(path):
    mm = tf.keras.models.load_model(path)
    mm.compile(
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'], 
    optimizer='adam'
    )

    return mm
  

def prediction_function(img): 
    
    #preprocess image, convert to 80 by 80 
    sample_image = cv.resize(np.array(img), (80, 80))
    
    cnn_model = load_prediction_model('tomato_prediction_model.h5')

    sample_image = np.expand_dims(sample_image, axis=0)
    print(sample_image.shape)
    prediction = cnn_model.predict(sample_image)
    output = np.argmax(prediction)
    print('Confidence : ', prediction[0][output] * 100 , '%')
    confidence = prediction[0][output] * 100
    print(output)
    return label_converter[output], confidence, 


if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=120)
    st.write("")

    # call prediction function 
    if st.button('Predict'):
        output, conf = prediction_function(image)


        # Display prediction
        st.write("Prediction: ", output)
        st.success('Confidence: ' + str(conf) + ' %' )
