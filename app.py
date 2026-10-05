import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Batman Rating Predictor",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gotham Dark Custom Styling
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
        font-size: 2.8rem;
        text-align: center;
        letter-spacing: -0.5px;
        margin-top: 10px;
    }
    .sub-title {
        color: #a0aec0;
        font-weight: 500;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* Prediction Result Banner */
    .result-banner {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 2px solid #f6ad55;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 0 25px rgba(246, 173, 85, 0.2);
        margin-top: 25px;
    }
    .result-score {
        font-size: 4.5rem;
        font-weight: 900;
        color: #f6ad55;
        margin: 12px 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #11141d !important;
        border-right: 1px solid #2d3748;
    }
    .sidebar-header {
        color: #f6ad55;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* Footer */
    .footer-text {
        color: #718096;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR: Project Context & Model Details
# ==========================================
with st.sidebar:
  st.title("🦇 Project Info")

  st.markdown('<div class="sidebar-header">🎯 Overview</div>', unsafe_allow_html=True)
  st.write("""
    This app uses a **Linear Regression** model trained on Batman film data from Kaggle.
    * **Target ($y$):** IMDb Rating
    * **Feature ($X$):** Release Year
    """)

  st.divider()

  st.markdown(
      '<div class="sidebar-header">⚙️ Training Steps</div>',
      unsafe_allow_html=True,
  )
  st.markdown("""
    1. Data ingestion & cleaning
    2. Baseline MAE computation
    3. Model fitting ($y = w \\cdot X + b$) via `scikit-learn`
    """)

  st.divider()

  st.markdown(
      '<div class="sidebar-header">📉 Performance Metrics</div>',
      unsafe_allow_html=True,
  )
  col_a, col_b = st.columns(2)
  col_a.metric("Baseline MAE", "0.7505")
  col_b.metric("Model MAE", "0.7183", delta="-4.3%")

  st.divider()

  st.markdown(
      '<div class="sidebar-header">📐 Equation</div>', unsafe_allow_html=True
  )
  st.latex(r"\text{IMDb} = 0.0226 \cdot \text{Year} - 38.1322")
  st.caption("Each passing year adds ~0.0226 points on average.")


# ==========================================
# MAIN PAGE: Focus on Prediction
# ==========================================
st.markdown(
    '<div class="main-title">🦇 Batman Movie Rating Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Enter a release year below to forecast its IMDb'
    " audience rating in real time.</div>",
    unsafe_allow_html=True,
)

# Centered Interactive Predictor
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
  year_input = st.number_input(
      "Select Movie Release Year:",
      min_value=1930,
      max_value=2050,
      value=2024,
      step=1,
  )

  # Model Weights
  w = 0.02262446
  b = -38.13222304

  # Calculate Linear Regression Prediction
  predicted_rating = w * year_input + b
  clamped_rating = min(max(predicted_rating, 1.0), 10.0)

  # Display Main Score Result
  st.markdown(
      f"""
    <div class="result-banner">
        <div style="font-size: 1.2rem; color: #e2e8f0; font-weight: 600;">Predicted IMDb Score for {year_input}</div>
        <div class="result-score">{clamped_rating:.2f} <span style="font-size: 2rem; color: #a0aec0;">/ 10</span></div>
        <div style="font-size: 0.9rem; color: #a0aec0;">Computed live via $y = 0.0226 \\times {year_input} - 38.1322$</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown(
    '<div class="footer-text">Streamlit • Scikit-Learn • GitHub'
    " Deployment</div>",
    unsafe_allow_html=True,
)
