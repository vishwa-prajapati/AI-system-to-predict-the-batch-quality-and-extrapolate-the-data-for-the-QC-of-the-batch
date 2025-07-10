
import streamlit as st
import pandas as pd
import joblib

# Load model and feature order
model = joblib.load("milk_quality_model.pkl")
model_features = joblib.load("model_features.pkl")  # List of features used in training

# Label mapping
label_map = {0: 'Bad', 1: 'Good', 2: 'Borderline'}

# Title
st.title("🥛 Milk Quality Prediction App (Preprocessed Inputs)")

st.markdown("Enter preprocessed (scaled) batch values:")

# Manually input key features (scaled values)
manual_inputs = {
    'pH_Level': st.number_input("pH Level", value=0.9731),
    'Fat_Content (%)': st.number_input("Fat Content (%)", value=-0.1172),
    'SNF (Solid-Not-Fat) (%)': st.number_input("SNF (%)", value=0.4882),
    'Microbial_Load (CFU/mL)': st.number_input("Microbial Load (CFU/mL)", value=0.1115),
    'Adulterant_Urea (ppm)': st.number_input("Urea (ppm)", value=-1.1477),
    'Adulterant_Starch (ppm)': st.number_input("Starch (ppm)", value=-1.0414),
    'Adulterant_Detergent (ppm)': st.number_input("Detergent (ppm)", value=0.9548),
    'Final_Product_Fat (%)': st.number_input("Final Product Fat (%)", value=1.6075),
    'Shelf_Life_Estimate (days)': st.number_input("Shelf Life Estimate (days)", value=1.4756),
    'QC_Score': st.number_input("QC Score", value=-0.5695),
}

st.markdown("All other features will be auto-filled with 0.")

if st.button("Predict Milk Quality"):
    # Build full input with manual + default 0s
    full_input = {feature: manual_inputs.get(feature, 0) for feature in model_features}
    
    # Convert to DataFrame
    input_df = pd.DataFrame([full_input])
    
    # Predict (no scaling needed)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Output
    st.success(f"🧾 Predicted Milk Quality: **{label_map[prediction]}**")

    st.subheader("📊 Prediction Probabilities")
    for i, label in label_map.items():
        st.write(f"- {label}: {probabilities[i]:.2%}")
