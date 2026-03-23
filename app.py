import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd

# Load model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Autonomous Fraud Detection", layout="wide")

# Header
st.title("💳 Autonomous Real-Time Financial Fraud Detection")
st.markdown("AI-powered fraud risk scoring with explainable predictions")
st.divider()

# Feature names
feature_names = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

st.subheader("🧾 Transaction Details")

# -------------------------------
# 🔹 Session State Initialization
# -------------------------------

if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * 30

cols = st.columns(4)

for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        st.session_state.inputs[i] = st.number_input(
            feature,
            value=st.session_state.inputs[i],
            key=f"input_{i}"
        )

st.divider()

# -------------------------------
# 🎯 Demo Transaction Loader
# -------------------------------

col_demo1, col_demo2 = st.columns(2)

with col_demo1:
    if st.button("🔴 Load Sample Fraud Case"):
        demo_values = [0.0] * 30
        demo_values[-1] = 5000  # High Amount example
        st.session_state.inputs = demo_values
        st.experimental_rerun()

with col_demo2:
    if st.button("🟢 Load Sample Legitimate Case"):
        demo_values = [0.0] * 30
        demo_values[-1] = 50  # Normal Amount example
        st.session_state.inputs = demo_values
        st.experimental_rerun()

inputs = st.session_state.inputs

# -------------------------------
# 🚀 Prediction Button
# -------------------------------

if st.button("🚀 Predict Fraud Risk", use_container_width=True):

    data = np.array(inputs).reshape(1, -1)
    df_input = pd.DataFrame(data, columns=feature_names)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    risk_score = round(probability * 100, 2)

    # -------------------------------
    # 🔍 Risk Assessment
    # -------------------------------

    st.subheader("🔍 Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠ Fraudulent Transaction Detected")
        else:
            st.success("✅ Transaction Appears Legitimate")

    with col2:
        st.metric("Risk Score", f"{risk_score}/100")

    st.markdown("### 📊 Fraud Probability")
    st.progress(float(probability))

    if risk_score > 70:
        st.warning("🚫 Recommended Action: Block Transaction")
    else:
        st.success("✅ Recommended Action: Allow Transaction")

    # -------------------------------
    # 🔎 Explainability Section
    # -------------------------------

    st.divider()
    st.subheader("🧠 Explainability (Top Influential Features)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values
    })

    feature_importance["Absolute Impact"] = feature_importance["Impact"].abs()

    top_features = feature_importance.sort_values(
        by="Absolute Impact", ascending=False
    ).head(5)

    # 📊 Horizontal Bar Chart
    st.markdown("### 📊 Feature Impact Visualization")

    chart_data = top_features.sort_values("Impact")
    st.bar_chart(chart_data.set_index("Feature")["Impact"])

    # 📌 Detailed Interpretation
    st.markdown("### 📌 Detailed Interpretation")

    for _, row in top_features.iterrows():
        if row["Impact"] > 0:
            st.write(
                f"• **{row['Feature']}** had a positive contribution, increasing fraud risk."
            )
        else:
            st.write(
                f"• **{row['Feature']}** had a negative contribution, reducing fraud risk."
            )

st.markdown("---")
st.markdown("Developed by Mangai | Real-Time ML with Explainable AI")