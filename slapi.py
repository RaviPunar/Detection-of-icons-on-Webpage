#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os

# Load the saved model
model = load_model('tl_model_v4.weights.best.h5')

# Define the function to classify an image
def classify_image(image_file, model):
    # Load the image
    img = image.load_img(image_file, target_size=(120, 120))
    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Make predictions
    preds = model.predict(x)
    # Get the class labels
    class_labels = sorted(os.listdir('icon/train'))[:43]
    # Get the predicted class
    pred = np.argmax(preds, axis=1)
    predicted_class = class_labels[pred[0]]
    return predicted_class, img

# Create the Streamlit app
def app():
    st.set_page_config(page_title='Detection of Icons using Webpage', page_icon=':icon:', layout='wide')
    st.title('Detection of Icons using Webpage')
    st.markdown('This app identifies an image using a Transfer Learning in pre-trained VGG16 model by making the last 2 layers as trainable and making the rest of the layers as non-trainable.')
    st.markdown('---')
    # Upload the image
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Classify the image and get the predicted class and the image
        predicted_class, img = classify_image(uploaded_file, model)
        # Display the predicted class and the image
        st.success(f'Predicted class: {predicted_class}')
        st.image(img, caption='Uploaded Image', width=120)
# Call the app() function
if __name__ == '__main__':
    app()


