import streamlit as st
import requests

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
    # Prepare file for FastAPI
    files = {"file": uploaded_file.getvalue()}
    headers = {"x-api-key": api_key}

    with st.spinner("Sending image to FastAPI model..."):
        try:
            response = requests.post(api_url, files=files, headers=headers, timeout=60)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction received!")

                st.write("### 🔍 Prediction Result")
                st.json(result)
                
            else:
                st.error(f"❌ Error from FastAPI server: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error("Error connecting to the FastAPI server.")
            st.write(e)
