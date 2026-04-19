import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Aerial Detection", layout="wide")

# Load model
model = load_model("bird_drone_model.keras")

# Sidebar
st.sidebar.title("⚙️ Settings")

model_type = st.sidebar.selectbox(
    "Classification Model",
    ["MobileNet", "Custom CNN"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for Aerial Surveillance AI Project")

# Title
st.title("🚁 Aerial Object Classification & Detection")

# Layout columns
col1, col2 = st.columns(2)

# Upload Section
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Prediction Section
with col2:
    st.subheader("📊 Results")

    if uploaded_file:
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        if prediction > confidence_threshold:
            label = "🚁 DRONE DETECTED"
            confidence = prediction
            color = "#ff4b4b"
        else:
            label = "🐦 BIRD DETECTED"
            confidence = 1 - prediction
            color = "#4CAF50"

        # Styled result box
        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:10px;
                background-color:{color}20;
                border:2px solid {color};
                text-align:center;">
                <h3>{label}</h3>
                <p>Confidence: {confidence*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Progress bar
        st.progress(float(confidence))