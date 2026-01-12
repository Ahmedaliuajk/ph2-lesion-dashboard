import streamlit as st
import cv2
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Skin Lesion Detection (PH2)",
    layout="centered"
)

st.title("Skin Lesion Detection System")
st.write("Machine Learning–based analysis using PH2 Dataset")

# ----------------------------
# Load trained model
# ----------------------------
with open("PH2_RF_Model.pkl", "rb") as f:
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

    features = {
        "Mean_Intensity": np.mean(lesion_pixels),
        "Std_Intensity":  np.std(lesion_pixels),
        "Area":           np.sum(mask > 0),
        "R_Mean":         np.mean(img[:,:,0][mask > 0]),
        "G_Mean":         np.mean(img[:,:,1][mask > 0]),
        "B_Mean":         np.mean(img[:,:,2][mask > 0]),
    }

    return features, mask

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Dermoscopic Image",
    type=["jpg", "png", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    features_dict, mask = extract_features(img)

    st.image(mask, caption="Generated Lesion Mask", use_column_width=True)

    if features_dict is None:
        st.warning("Lesion could not be detected clearly.")
    else:
        # Convert to DataFrame (same format as training)
        feature_df = pd.DataFrame([features_dict])

        # Prediction
        pred = model.predict(feature_df)[0]
        prob = model.predict_proba(feature_df)[0]

        if pred == 1:
            st.error(f"Lesion Detected (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"No Lesion Detected (Confidence: {prob[0]*100:.2f}%)")

        # Display features clearly
        st.subheader("Extracted Features")
        st.dataframe(feature_df)
