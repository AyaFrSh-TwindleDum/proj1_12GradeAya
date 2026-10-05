import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Batman Movie Rating Predictor",
    page_icon="🦇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Dark Gotham & Gold Custom Styling
st.markdown(
    """
    <style>
    /* Main Dark Theme Background */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        color: #f6ad55;
        font-weight: 900;
        font-size: 2.5rem;
        text-align: center;
        letter-spacing: -0.5px;
        margin-top: 10px;
    }
    .sub-title {
        color: #cbd5e0;
        font-weight: 500;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Dark Gotham Cards */
    .gotham-card {
        background-color: #171923;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #2d3748;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .gotham-header {
        color: #f6ad55;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metrics Styling */
    [data-testid="stMetric"] {
        background-color: #1a202c !important;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3748;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #f6ad55 !important;
        font-weight: 800 !important;
    }

    /* Prediction Result Banner */
    .result-banner {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 2px solid #f6ad55;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 20px rgba(246, 173, 85, 0.15);
        margin-top: 20px;
    }
    .result-score {
        font-size: 3.2rem;
        font-weight: 900;
        color: #f6ad55;
        margin: 8px 0;
    }
    
    /* Footer */
    .footer-text {
        color: #718096;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 35px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title & Subtitle
st.markdown(
    '<div class="main-title">🦇 Batman Rating Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Linear Regression Model trained on Kaggle Batman'
    " Film Data</div>",
    unsafe_allow_html=True,
)

# Overview Section
st.markdown(
    """
<div class="gotham-card">
    <div class="gotham-header">🎯 Project Scope</div>
    An end-to-end Machine Learning web app predicting movie performance using simple Linear Regression.
    <br><br>
    • <b>Target Variable ($y$):</b> IMDb Audience Score (<code>Imdb Rating</code>)<br>
    • <b>Predictor Feature ($X$):</b> Release Year (<code>Year</code>)
</div>
""",
    unsafe_allow_html=True,
)

# Process Section
st.markdown(
    """
<div class="gotham-card">
    <div class="gotham-header">⚙️ Training Methodology</div>
    1. <b>Data Pipeline:</b> Downloaded and cleaned Batman film records from Kaggle.<br>
    2. <b>Baseline Benchmark:</b> Computed mean target value to set initial MAE loss.<br>
    3. <b>Model Fitting:</b> Derived $y = w \\cdot X + b$ parameters using <code>scikit-learn</code>.
</div>
""",
    unsafe_allow_html=True,
)

# Metrics Grid
col1, col2 = st.columns(2)
with col1:
  st.metric(
      label="📉 BASELINE MAE",
      value="0.7505",
      help="Mean baseline error predicting the average rating.",
  )

with col2:
  st.metric(
      label="🚀 MODEL MAE",
      value="0.7183",
      delta="-4.3% Error",
      delta_color="normal",
      help="Trained linear regression mean absolute error.",
  )

# Model Formula Box
st.info("""
**Learned Model Equation:**
$$\\text{IMDb Rating} = 0.0226 \\times \\text{Year} - 38.1322$$

* **Slope ($w \\approx 0.0226$):** Each year adds roughly $+0.0226$ rating points on average.
* **Intercept ($b \\approx -38.1322$):** Mathematical constant offset.
""")

st.divider()

# Interactive Predictor Section
st.markdown("### 🎯 Predict a Film's Rating")

year_input = st.number_input(
    "Enter Movie Release Year:",
    min_value=1930,
    max_value=2050,
    value=2024,
    step=1,
)

# Weights
w = 0.02262446
b = -38.13222304

# Compute prediction
predicted_rating = w * year_input + b
clamped_rating = min(max(predicted_rating, 1.0), 10.0)

# Display Result Card
st.markdown(
    f"""
<div class="result-banner">
    <div style="font-size: 1.1rem; color: #e2e8f0;">Predicted IMDb Score ({year_input})</div>
    <div class="result-score">{clamped_rating:.2f} <span style="font-size: 1.5rem; color: #a0aec0;">/ 10</span></div>
    <div style="font-size: 0.85rem; color: #a0aec0;">Formula: $y = 0.0226 \\times {year_input} - 38.1322$</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer-text">Streamlit • Scikit-Learn • GitHub Deployment</div>',
    unsafe_allow_html=True,
)
