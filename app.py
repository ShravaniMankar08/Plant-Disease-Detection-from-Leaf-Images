import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from disease_info import DISEASE_INFO  # ← Ensure this file exists in the same folder
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
# --- Configuration ---
MODEL_PATH = "model/plant_disease_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = sorted(os.listdir("dataset/train"))  # Automatically grabs class names

# --- Load Model Once ---
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# --- Preprocess Image ---
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# --- Streamlit UI ---
#st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and let the AI predict the disease.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        try:
            # Predict
            processed = preprocess_image(image)
            predictions = model.predict(processed)

            class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            label = CLASS_NAMES[class_index]

            # Show prediction
            st.success(f"✅ Prediction: **{label}** ({confidence*100:.2f}% confidence)")

            # Show disease info
            info = DISEASE_INFO.get(label, {})
            if info:
                st.markdown("### 🧠 Disease Details")
                st.markdown(f"**📝 Description:** {info['description']}")
                st.markdown(f"**🔍 Symptoms:** {info['symptoms']}")
                st.markdown(f"**💊 Treatment:** {info['treatment']}")
            else:
                st.info("ℹ️ No additional information available for this disease.")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
