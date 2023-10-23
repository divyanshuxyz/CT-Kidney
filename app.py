import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.models.load_model("model.h5")
st.title("CT KIDNEY")
st.subheader("NORMAL, CYST, TUMOR, STONE DETECTION")
st.subheader("Image Classification ")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
labels={0: 'CYST', 1: 'NORMAL', 2: 'STONE', 3: 'TUMOR'}
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the image
    img = image.load_img(uploaded_image)
    resize = tf.image.resize(img, (150,150))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    max_index = np.argmax(yhat)
    decoded_predictions=labels[max_index]
    if decoded_predictions!='NORMAL':
         #st.subheader("The person has",decoded_predictions,"in his kidney")
         st.subheader(f"The person has {decoded_predictions} in his kidney.")
    else:
         st.subheader("The kidney is NORMAL")
#to run the code enter
#streamlit run app.py    