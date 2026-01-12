import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Skin Lesion Detection (PH2)",
    layout="centered"
)

st.title("Skin Lesion Detection System")
st.write("PH2 Dataset Based Machine Learning Model")

# ----------------------------
# Load trained model
# ----------------------------
with open('PH2_RF_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------------------
# Feature extraction function
# ----------------------------
def extract_features(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    lesion_pixels = gray[mask > 0]

    if len(lesion_pixels) < 100:
        return None, mask

    features = np.array([[
        np.mean(lesion_pixels),
        np.std(lesion_pixels),
        np.sum(mask > 0),
        np.mean(img[:,:,0][mask > 0]),
        np.mean(img[:,:,1][mask > 0]),
        np.mean(img[:,:,2][mask > 0])
    ]])

    return features, mask

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload dermoscopic image",
    type=['jpg', 'png', 'bmp']
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    features, mask = extract_features(img)

    st.image(mask, caption="Generated Lesion Mask", use_column_width=True)

    if features is None:
        st.warning("Lesion not detected clearly. Please use PH2-like dermoscopic images.")
    else:
        # Prediction
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        if pred == 1:
            st.error(f"Lesion Detected (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"No Lesion Detected (Confidence: {prob[0]*100:.2f}%)")

        # Debug info (optional)
        st.write("Extracted Features:", features)
