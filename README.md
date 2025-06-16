🌿 Plant Disease Detection Web App
A deep learning–based web application built with Streamlit that detects plant leaf diseases from images and provides helpful information including symptoms and treatments.

🚀 Demo
Upload a leaf image and the AI will predict the disease:
![image](https://github.com/user-attachments/assets/e52b5ab8-c368-4cb1-b755-857c165ee05a)

🧠 Features
🔍 Image-based Disease Prediction using a CNN model
📑 Detailed Explanation of disease (description, symptoms, treatment)
🖼️ Supports .jpg, .jpeg, .png images
⚙️ Built with TensorFlow, OpenCV, Keras, Streamlit
📦 Easily deployable to Streamlit Cloud

🧪 Model Information
Architecture: Convolutional Neural Network (CNN)

Input size: 128x128 RGB images

Trained on: Custom or Kaggle leaf disease dataset( link:https://www.kaggle.com/datasets/emmarex/plantdisease)

Classes:
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Septoria Leaf Spot
Target Spot
Tomato Yellow Leaf Curl Virus
Tomato Mosaic Virus
Healthy

🔧 Installation & Run Locally
1. Clone the Repo
bash
git clone https://github.com/yourusername/Plant_Disease_Detection.git
cd Plant_Disease_Detection
2. Install Requirements
bash
pip install -r requirements.txt
3. Run the App
bash
streamlit run app.py

🔍 Example Prediction Output
Upload a leaf image
App predicts: Early Blight (94.23% confidence)
It shows:
📄 Description
🚨 Symptoms
💊 Suggested Treatment

