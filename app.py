import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="ARX System Model", layout="centered")

st.title("Electrical Dynamic System Predictor")
st.write("ARX-based machine learning model for system identification")

# Load trained model
model = joblib.load("arx_system_model.pkl")

# Safety check (ensures model is fitted)
_ = model.coef_

st.subheader("Enter Past Input and Output Values")

y_k_1 = st.number_input("Previous Output y(k-1)", value=120.0)
y_k_2 = st.number_input("Previous Output y(k-2)", value=118.0)
u_k_1 = st.number_input("Previous Input u(k-1)", value=60.0)
u_k_2 = st.number_input("Previous Input u(k-2)", value=58.0)

if st.button("Predict Output"):
    X_new = np.array([[y_k_1, y_k_2, u_k_1, u_k_2]])
    y_pred = model.predict(X_new)
    st.success(f"Predicted Output: {y_pred[0]:.2f}")
