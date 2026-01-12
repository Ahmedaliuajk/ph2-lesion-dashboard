import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

st.set_page_config(page_title="PH2 Lesion Detection", layout="centered")

# Load model
with open('PH2_RF_Model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Skin Lesion Detection System")
st.write("Upload a dermoscopic image")

uploaded_file = st.file_uploader("Choose an image", type=['jpg','png','bmp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    st.image(mask, caption="Generated Mask", use_column_width=True)

    lesion_pixels = gray[mask > 0]

    if len(lesion_pixels) < 50:
        st.warning("Lesion not clearly detected")
    else:
        features = np.array([[
            np.mean(lesion_pixels),
            np.std(lesion_pixels),
            np.sum(mask > 0),
            np.mean(img[:,:,0][mask > 0]),
            np.mean(img[:,:,1][mask > 0]),
            np.mean(img[:,:,2][mask > 0])
        ]])

        pred = model.predict(features)[0]

        if pred == 1:
            st.error("Lesion Detected")
        else:
            st.success("No Lesion Detected")
