import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="CNN Image Classifier", layout="centered")

st.title("🖼️ CNN Image Classifier (FastAPI + ngrok)")
st.write("Upload an image and provide your FastAPI (ngrok) URL.")

# User inputs their ngrok URL
api_url = st.text_input("Enter your FastAPI /predict URL:", placeholder="https://xxxx.ngrok-free.app/predict")

# API key input
api_key = st.text_input("Enter your API Key:", type="password")

# User uploads image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and api_url and api_key:

    # Display the uploaded picture
    st.write("### 📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    # Prepare file for FastAPI
    files = {"file": uploaded_file.getvalue()}
    headers = {"x-api-key": api_key}

    with st.spinner("Sending image to FastAPI model..."):
        try:
            response = requests.post(api_url, files=files, headers=headers, timeout=60)

            if response.status_code == 200:
                result = response.json()

                # Extract what you want
                disease_name = result.get("prediction_name")
                confidence_scores = result.get("confidence_percentages")

                # Get highest confidence
                confidence = confidence_scores[0] if isinstance(confidence_scores, list) else None

                st.success("Prediction received!")

                # Clean formatted result
                st.write("### 🧪 Classification Result")
                st.write(f"**Disease:** {disease_name}")
                st.write(f"**Confidence:** {confidence:.2f}%")

            else:
                st.error(f"❌ Error from FastAPI server: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error("Error connecting to the FastAPI server.")
            st.write(e)
