import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model from Google Drive if not already present
MODEL_PATH = "potato_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1L-HX6Otifp67grwfJaCKRFI93ZLBHMmH"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (update according to your dataset)
class_names = ['Early_Blight', 'Late_Blight', 'Healthy']
IMAGE_SIZE = 256

st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image and detect the disease using AI.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence}%")
