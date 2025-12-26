import streamlit as st
import requests
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title="CNN Image Classifier", layout="centered")
st.title("🖼️ CNN Image Classifier (FastAPI + ngrok)")
st.write("Upload an image and provide your FastAPI (ngrok) URL and API key.")

# --- User Inputs ---
api_url = st.text_input(
    "Enter your FastAPI /predict URL:",
    placeholder="https://xxxx.ngrok-free.app/predict"
)

api_key = st.text_input("Enter your API Key:", type="password")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --- Main Logic ---
if uploaded_file and api_url and api_key:
    # Display uploaded image
    st.write("### 📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    # --- Prepare file for FastAPI ---
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    headers = {"x-api-key": api_key}

    with st.spinner("Sending image to FastAPI model..."):
        try:
            response = requests.post(api_url, files=files, headers=headers, timeout=60)

            if response.status_code == 200:
                result = response.json()

                # --- Extract predicted class and confidence ---
                prediction_index = result.get("prediction_index", 0)
                prediction_name = result.get("prediction_name", "Unknown")
                confidence_percentages = result.get("confidence_percentages", [])

                # Pick confidence of the predicted class
                if confidence_percentages and prediction_index < len(confidence_percentages):
                    confidence = confidence_percentages[prediction_index]
                else:
                    confidence = 0

                st.success("✅ Prediction received!")

                # --- Display results ---
                st.write("### 🧪 Classification Result")
                st.write(f"**Disease:** {prediction_name}")
                st.write(f"**Confidence:** {confidence:.2f}%")

            else:
                st.error(f"❌ Error from FastAPI server: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error("⚠️ Could not connect to the FastAPI server.")
            st.write(e)
