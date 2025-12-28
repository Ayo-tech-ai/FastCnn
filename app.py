
import streamlit as st
import requests
from PIL import Image
import base64
import json
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
    "Upload a **clear image of a cassava leaf**, then click **Analyze Image** to begin."
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

analyze_btn = st.button("🔍 Analyze Image")

# --------------------------------------------------
# MAIN PIPELINE (ONLY RUNS ON BUTTON CLICK)
# --------------------------------------------------
if analyze_btn:

    # Basic guard checks
    if not uploaded_file:
        st.warning("Please upload an image first.")
        st.stop()

    if not (api_url and cnn_api_key and openai_api_key):
        st.warning("Please provide all required keys and URLs.")
        st.stop()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=320)

    # Convert image to base64 (for OpenAI)
    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    client = OpenAI(api_key=openai_api_key)

    # --------------------------------------------------
    # STEP 0: CASSAVA VALIDATION (HARD SAFE GATE)
    # --------------------------------------------------
    with st.spinner("Validating image..."):
        try:
            validation_response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": """
You are an agricultural vision expert.

Determine whether the image clearly shows **cassava leaves (Manihot esculenta)**.

Respond ONLY in valid JSON using exactly this format:

{
  "is_cassava": true or false,
  "reason": "short explanation"
}

If the image is unclear, not a plant, or not cassava, return is_cassava as false.
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

            validation_text = validation_response.output_text
            validation_result = json.loads(validation_text)

            is_cassava = validation_result.get("is_cassava", False)

        except Exception:
            st.error(
                "This picture does not look like a cassava image, "
                "or the picture is not clear enough. "
                "Please upload a clear picture of a cassava leaf."
            )
            st.stop()

    # HARD STOP IF VALIDATION FAILS
    if not is_cassava:
        st.error(
            "❌ This picture does not look like a cassava image, "
            "or the picture is not clear enough.\n\n"
            "Please upload a **clear picture of a cassava leaf**."
        )
        st.stop()

    # --------------------------------------------------
    # STEP 1: CNN DISEASE DETECTION
    # --------------------------------------------------
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    headers = {"x-api-key": cnn_api_key}

    with st.spinner("Analyzing disease..."):
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

        except Exception:
            st.error("Connection error. Please try again.")
            st.stop()

    prediction_index = result.get("prediction_index", 0)
    disease_name = result.get("prediction_name", "Unknown")
    confidence_percentages = result.get("confidence_percentages", [])

    confidence = 0.0
    if isinstance(confidence_percentages, list) and confidence_percentages:
        flat_conf = (
            confidence_percentages[0]
            if isinstance(confidence_percentages[0], list)
            else confidence_percentages
        )
        if prediction_index < len(flat_conf):
            confidence = flat_conf[prediction_index]

    # --------------------------------------------------
    # STEP 2: ADVISORY GENERATION
    # --------------------------------------------------
    with st.spinner("Preparing advisory report..."):
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
An AI system analyzed an image of cassava leaves and identified the disease as
**{disease_name}** with a confidence of **{confidence:.2f}%**.

Provide a **structured, farmer-friendly advisory report** including:
- Visible signs and symptoms
- Cause of the disease
- How it spreads
- Symptoms at different stages
- Preventive measures suitable for African farmers
- Practical control actions
- What farmers should monitor in the future

Do NOT use conversational language or self-references.
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
                "Disease detected successfully, but advisory guidance "
                "could not be generated."
            )

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    st.success("Analysis Complete")

    st.write("### 🧪 Detection Result")
    st.write(f"**Identified Condition:** {disease_name}")
    st.write(f"**Confidence Level:** {confidence:.2f}%")

    st.write("### 🌾 Advisory & Recommendations")
    st.markdown(explanation)
