import streamlit as st
import requests
from PIL import Image
import base64
from openai import OpenAI

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    layout="centered"
)

st.title("🌱 AI Crop Disease Detection & Advisory")
st.write(
    "Upload an image of a crop leaf to receive disease detection and practical farming guidance."
)

# --------------------------------------------------
# User inputs
# --------------------------------------------------
api_url = st.text_input(
    "FastAPI Prediction URL",
    placeholder="https://xxxx.ngrok-free.app/predict"
)

cnn_api_key = st.text_input(
    "System Access Key",
    type="password"
)

openai_api_key = st.text_input(
    "AI Advisory Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
if uploaded_file and api_url and cnn_api_key and openai_api_key:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=320)

    # --------------------------------------------------
    # Step 1: Disease Detection (internal)
    # --------------------------------------------------
    # --- FIX: Send image in proper format for FastAPI ---
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    headers = {"x-api-key": cnn_api_key}

    with st.spinner("Analyzing image..."):
        try:
            response = requests.post(
                api_url,
                files=files,
                headers=headers,
                timeout=60
            )

            if response.status_code != 200:
                st.error("Unable to analyze image at the moment.")
                st.stop()

            result = response.json()

        except Exception as e:
            st.error("Connection error. Please try again.")
            st.stop()

    # --------------------------------------------------
    # Correct confidence extraction (robust version)
    # --------------------------------------------------
    prediction_index = result.get("prediction_index", 0)
    disease_name = result.get("prediction_name", "Unknown")
    confidence_percentages = result.get("confidence_percentages", [])

    confidence = 0.0

    if isinstance(confidence_percentages, list) and len(confidence_percentages) > 0:
        # If it's a nested list, flatten it
        if isinstance(confidence_percentages[0], list):
            flat_conf = confidence_percentages[0]
        else:
            flat_conf = confidence_percentages

        # Pick the confidence of the predicted class
        if prediction_index < len(flat_conf):
            confidence = flat_conf[prediction_index]

    # --------------------------------------------------
    # Step 2: Explanation & Guidance (internal)
    # --------------------------------------------------
    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    client = OpenAI(api_key=openai_api_key)

    with st.spinner("Preparing insights and recommendations..."):
        try:
            ai_response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""
An AI system has analyzed an image of a cassava plant and identified
the disease as **{disease_name}** with a confidence of **{confidence:.2f}%**.

Please provide a clear, farmer-friendly explanation that includes:
- What is visible in the image
- The likely cause of the disease
- How the disease spreads
- Symptoms relevant to what is seen
- Preventive measures
- Practical actions the farmer can take

Keep the explanation natural and helpful.
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

            explanation = ai_response.output_text

        except Exception:
            explanation = (
                "The disease was detected successfully, but additional guidance "
                "could not be generated at this time."
            )

    # --------------------------------------------------
    # Unified output (ONE FLOW)
    # --------------------------------------------------
    st.success("Analysis Complete")

    st.write("### 🧪 Detection Result")
    st.write(f"**Identified Condition:** {disease_name}")
    st.write(f"**Confidence Level:** {confidence:.2f}%")

    st.write("### 🌾 Advisory & Recommendations")
    st.markdown(explanation)

else:
    st.info("Please upload an image and provide all required details to begin.")
