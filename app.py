import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Batman Movie Rating Predictor",
    page_icon="🦇",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Relaxing & Modern Visual Styling (Soft Violet / Lavender Palette)
st.markdown(
    """
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Title Styling */
    .main-title {
        color: #2d3748;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-title {
        color: #5a67d8;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards & Containers */
    .custom-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
    }
    .custom-card-header {
        color: #4c51bf;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Metric Box Customization */
    [data-testid="stMetric"] {
        background-color: #f7fafc;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #edf2f7;
    }
    [data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #2b6cb0 !important;
        font-weight: 700 !important;
    }

    /* Target Result Display */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        margin-top: 20px;
    }
    .result-score {
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -1px;
    }
    
    /* Footer Note */
    .footer-note {
        color: #a0aec0;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 30px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 1. Header Section
st.markdown(
    '<div class="main-title">🦇 Batman Movie Rating Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Interactive Linear Regression Machine Learning'
    " Project</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="custom-card">
    <div class="custom-card-header">🎯 Project Overview</div>
    Welcome to the interactive predictive application! This project demonstrates an end-to-end Machine Learning pipeline using a dataset of Batman movies from Kaggle.<br><br>
    <ul>
        <li><b>Target Variable ($y$):</b> Audience score on IMDb (<code>Imdb Rating</code>)</li>
        <li><b>Feature Predictor ($X$):</b> Release year of the movie (<code>Year</code>)</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# 2. Model Training & Evaluation Section
st.markdown(
    """
<div class="custom-card">
    <div class="custom-card-header">📚 Training Process & Loss Performance</div>
    The model was trained in a Jupyter research notebook (Colab) following these core steps:
    <ol>
        <li><b>Data Ingestion:</b> Loaded the Batman film dataset from Kaggle and handled missing values.</li>
        <li><b>Feature Extraction:</b> Extracted <code>Year</code> ($X$) and <code>IMDb Rating</code> ($y$) as NumPy numerical vectors.</li>
        <li><b>Baseline Model:</b> Calculated constant mean predictions to set an initial baseline loss benchmark.</li>
        <li><b>Linear Regression:</b> Fitted the optimal linear equation $y = w \cdot X + b$ using <code>scikit-learn</code> to minimize error.</li>
    </ol>
</div>
""",
    unsafe_allow_html=True,
)

# Metrics Grid
col1, col2 = st.columns(2)
with col1:
  st.metric(
      label="📉 Baseline Loss (MAE)",
      value="0.7505",
      help="Average error when predicting solely using the target mean score.",
  )

with col2:
  st.metric(
      label="🚀 Model Loss (MAE)",
      value="0.7183",
      delta="-4.3% Error Reduction",
      delta_color="normal",
      help="Average error achieved by the trained Linear Regression model.",
  )

# Mathematical Equation Summary
st.info("""
**Linear Regression Equation:**
$$\\text{IMDb Rating} = 0.0226 \\times \\text{Year} - 38.1322$$

* **Slope ($w \\approx 0.0226$):** On average, each passing year is associated with a 0.0226 increase in IMDb rating (showing a slight upward rating trend over time).
* **Intercept ($b \\approx -38.1322$):** The theoretical y-intercept offset of the linear model.
""")

st.divider()

# 3. Interactive Prediction Section
st.subheader("🎯 Live Interactive Simulator")
st.write(
    "Select or type any movie release year below to generate a real-time"
    " prediction from the trained model:"
)

# User Input Widget
year_input = st.number_input(
    "Select Release Year (Year):",
    min_value=1930,
    max_value=2050,
    value=2024,
    step=1,
)

# Learned Weights from Training
w = 0.02262446
b = -38.13222304

# Calculate Linear Regression Prediction
predicted_rating = w * year_input + b
clamped_rating = min(max(predicted_rating, 1.0), 10.0)

# Beautiful Score Card Output
st.markdown(
    f"""
<div class="result-card">
    <div style="font-size: 1.1rem; opacity: 0.9;">Predicted IMDb Rating for {year_input}</div>
    <div class="result-score">{clamped_rating:.2f} <span style="font-size: 1.5rem; opacity: 0.8;">/ 10</span></div>
    <div style="font-size: 0.85rem; opacity: 0.85;">Calculated via $y = 0.0226 \\times {year_input} - 38.1322$</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer-note">Built with Python, Scikit-Learn, and Streamlit'
    " 🦇</div>",
    unsafe_allow_html=True,
)
