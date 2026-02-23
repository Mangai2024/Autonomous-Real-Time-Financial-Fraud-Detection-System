# 💳 Autonomous Real-Time Financial Fraud Detection System

An end-to-end machine learning system for detecting financial fraud in real time, integrated with explainable AI techniques and deployed as a cloud-based interactive web application.

---

## 🚀 Project Overview

This project presents an autonomous fraud detection system that:

- Detects fraudulent transactions in real time
- Assigns dynamic fraud risk scores
- Provides transparent explanations using SHAP
- Simulates fraud and legitimate scenarios
- Deploys as a public cloud-hosted application

The system combines supervised machine learning with model explainability to support transparent and reliable financial decision-making.

---

## 🧠 Key Features

### 🔹 Real-Time Fraud Prediction
- Predicts transaction fraud probability
- Generates dynamic risk score (0–100)
- Provides automated decision logic (Allow / Block)

### 🔹 Explainable AI Integration
- Uses SHAP (TreeExplainer) for model transparency
- Displays top 5 influential features
- Visualizes feature impact using bar charts
- Provides human-readable interpretation

### 🔹 Interactive Demo Scenarios
- Manual transaction input
- Fraud example simulation
- Legitimate transaction simulation

### 🔹 Cloud Deployment
- Built with Streamlit
- Deployed on Hugging Face Spaces
- Public access for demonstration

---

## 🏗️ System Architecture

1. Transaction Input Interface
2. Feature Engineering (Manual Input)
3. Random Forest Model Inference
4. Risk Scoring Engine
5. SHAP Explainability Module
6. Decision Recommendation Logic
7. Cloud Deployment (Hugging Face)

---

## 📊 Model Details

- Algorithm: Random Forest Classifier
- Dataset: Credit Card Fraud Detection Dataset
- Features: Time, V1–V28, Amount
- Target: Fraud / Non-Fraud
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score
  - Fraud detection recall optimization

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- SHAP
- NumPy
- Pandas
- Streamlit
- Hugging Face Spaces

---

## 📈 Risk Scoring Logic

The fraud probability returned by the model is converted into a risk score:


Risk Score = Fraud Probability × 100


Decision Threshold:
- Risk Score > 70 → Block Transaction
- Risk Score ≤ 70 → Allow Transaction

---

## 🔎 Explainability Approach

SHAP (SHapley Additive exPlanations) is used to:

- Measure feature-level contribution
- Identify top influential transaction attributes
- Provide directional impact (increase/decrease fraud risk)
- Enhance model transparency

This ensures interpretability and supports trustworthy AI decision-making.

---

## 🌐 Live Application

Public Deployment Link:
👉 (Add your Hugging Face URL here)

---

## 📌 Future Enhancements

- Drift detection monitoring
- Continual learning simulation
- FastAPI backend integration
- Dashboard-based monitoring system
- Authentication & secure access layer

---

## 👩‍💻 Author

Developed by Mangai  
Real-Time ML | Explainable AI | Data Science Portfolio Project

---
