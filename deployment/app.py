import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title('Adidas and Nike shoes predictions')

# import the model

model = load_model('cnn.h5')

# define the preprocessing function

def preprocess_image(image):
    image = image.resize((242, 242))  
    image = image.convert("RGB")  
    image = np.array(image)  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# define the prediction function

def prediction(image):
    preprocessed_image = preprocess_image(image)
    classes = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(classes)
    class_labels = ['adidas', 'nike']
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# file uploader

uploaded_file = st.file_uploader("Upload picture to predict")

# result

if st.button('Predict'):
    if uploaded_file is None:
        st.write('Please upload a leaf picture first.')
    else:
        image = Image.open(uploaded_file)
        result = prediction(image)
        st.write('This leaf belongs to the {} class.'.format(result))