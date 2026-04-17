import streamlit as st
import pickle
import numpy as np
from PIL import Image
import time

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ----------------- Custom CSS (Animation + Style) -----------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff416c;
        transform: scale(1.05);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown('<div class="title">🤖 ML Prediction App</div>', unsafe_allow_html=True)
st.write("### Enter Input Values Below 👇")

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    with open("model.pkl(3)", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------------- Inputs -----------------
st.subheader("📊 Input Features")

# Change number of inputs based on your model
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

features = np.array([[feature1, feature2, feature3, feature4]])

# ----------------- Prediction Button -----------------
if st.button("🚀 Predict"):
    
    with st.spinner("Predicting..."):
        time.sleep(1.5)
        prediction = model.predict(features)

    st.success(f"✅ Prediction: {prediction[0]}")

    st.balloons()

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
