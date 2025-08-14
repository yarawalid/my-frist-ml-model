
import streamlit as st
import joblib
import numpy as np
import os

# ===================== Load the Model =====================
# استخدام المسار النسبي بدلاً من المسار المطلق
model_path = os.path.join(os.path.dirname(__file__), "kmeans_model.pkl")
model = joblib.load(model_path)

# ===================== Custom Feature Names =====================
feature_names = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

# ===================== Page Config =====================
st.set_page_config(
    page_title="💖 Cute ML Model",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===================== CSS for Pink Style =====================
st.markdown("""
    <style>
        .main {
            background-color: #fff0f5;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #d63384;
            text-align: center;
            font-family: 'Comic Sans MS', cursive;
        }
        .stButton>button {
            background-color: #ff69b4;
            color: white;
            border-radius: 12px;
            font-size: 18px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff1493;
        }
        .sidebar .sidebar-content {
            background-color: #ffe6f0;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== Title =====================
st.title("🌸 My Cute ML Model 🌸")
st.write("Enter your lovely data below and I'll give you a magical prediction ✨")

# ===================== Inputs =====================
st.subheader("💌 Enter Your Data")
input_values = []
for name in feature_names:
    val = st.number_input(f"{name} 💖", value=0.0, step=1.0)  # float type for all
    input_values.append(val)

# ===================== Prediction =====================
if st.button("🔮 Predict Now"):
    input_data = np.array([input_values])
    prediction = model.predict(input_data)
    st.success(f"🌟 Prediction Result: {prediction[0]} 🌟")

# ===================== Sidebar Info =====================
st.sidebar.header("🌷 About")
st.sidebar.write("""
- This model was pre-trained 💕
- Deployed with a lovely pink theme 🌸
- Change the values and see the magic happen 💫
""")


