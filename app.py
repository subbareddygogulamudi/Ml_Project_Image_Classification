import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2

# Load trained model
with open("rf_pipe.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Image Classification", layout="centered")

st.title("Image Classification on Blood Cells")
st.write("Upload an image to classify it ")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(image, caption="Uploaded Image")


    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.resize(img, (64, 64))

    # Flatten image (IMPORTANT for KNN)
    img_flat = img.reshape(1, -1)

    # Prediction
    predicted_label = model.predict(img_flat)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_label}")