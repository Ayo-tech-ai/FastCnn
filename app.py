import streamlit as st
import requests
from PIL import Image
import base64
from openai import OpenAI

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    layout="centered"
)

st.title("🌱 AI Crop Disease Detection & Advisory System")
st.write(
    """
    This system combines:
    - **CNN-based disease classification** (primary model)
    - **GPT-4 Vision** for explanation, symptoms, and preventive guidance
    """
)

# --------------------------------------------------
# User inputs
# --------------------------------------------------
api_url = st.text_input(
    "Enter your FastAPI /predict URL:",
    placeholder="https://xxxx.ngrok-free.app/predict"
)

cnn_api_key = st.text_input(
    "Enter your CNN API Key:",
    type="password"
)

openai_api_key = st.text_input(
    "Enter your OpenAI API Key:",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload an image of the crop/leaf",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file and api_url and cnn_api_key and openai_api_key:

    # Display uploaded image
    st.write("### 📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    # --------------------------------------------------
    # 1️⃣ CNN PREDICTION (Primary Model)
    # --------------------------------------------------
    files = {"file": uploaded_file.getvalue()}
    headers = {"x-api-key": cnn_api_key}

    with st.spinner("🔬 Running CNN disease classification..."):
        try:
            response = requests.post(
                api_url,
                files=files,
                headers=headers,
                timeout=60
            )

            if response.status_code != 200:
                st.error("❌ Error from CNN server")
                st.write(response.text)
                st.stop()

            cnn_result = response.json()

        except Exception as e:
            st.error("❌ Could not connect to CNN backend")
            st.write(e)
            st.stop()

    # Extract CNN output
    disease_name = cnn_result.get("prediction_name")
    confidence_scores = cnn_result.get("confidence_percentages")
    confidence = (
        confidence_scores[0]
        if isinstance(confidence_scores, list)
        else None
    )

    # Display CNN result
    st.success("✅ CNN Prediction Complete")
    st.write("### 🧪 CNN Classification Result")
    st.write(f"**Detected Disease:** {disease_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # --------------------------------------------------
    # 2️⃣ GPT-4V EXPLANATION (Secondary Model)
    # --------------------------------------------------
    st.write("### 🧠 AI Explanation & Farmer Guidance")

    # Encode image for GPT-4V
    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    client = OpenAI(api_key=openai_api_key)

    with st.spinner("📖 Generating explanation and agricultural guidance..."):
        try:
            gpt_response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""
A CNN model has detected the crop disease **{disease_name}**
with a confidence of **{confidence:.2f}%**.

You are an agricultural assistant.

Please provide:
1. A brief description of what you observe in the image
2. The likely causative agent of this disease
3. How the disease spreads
4. Symptoms that match what is visible in the image
5. Preventive measures
6. Practical actions a farmer can take

Do NOT change or dispute the CNN prediction.
Explain in clear, farmer-friendly language.
"""
                            },
                            {
                                "type": "input_image",
                                "image_url": image_data_url
                            }
                        ]
                    }
                ]
            )

            explanation = gpt_response.output_text

            st.markdown(explanation)

        except Exception as e:
            st.error("❌ GPT-4V explanation failed")
            st.write(e)

else:
    st.info("⬆️ Upload an image and provide all required URLs and API keys.")
