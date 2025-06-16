import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
print("started")
# --- CONFIG ---
MODEL_PATH = "model/plant_disease_model.h5"
IMG_SIZE = (128, 128)  # Same as used during training
CLASS_NAMES = sorted(os.listdir("dataset/train"))  # Assumes same folder structure used

# --- FUNCTION ---
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, 128, 128, 3)
    return image

def predict(image_path):
    model = load_model(MODEL_PATH)
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    label = CLASS_NAMES[class_index]
    
    print(f"[✅] Prediction: {label} ({confidence * 100:.2f}% confidence)")

# --- RUN FROM TERMINAL ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[❌] Error: File not found - {args.image_path}")
    else:
        predict(args.image_path)
