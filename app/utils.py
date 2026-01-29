"""
Utility functions for PCOS Streamlit Application
"""

import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import sys
import os

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from predict import PCOSPredictor


@st.cache_data
def load_processed_data():
    """Load the processed PCOS dataset"""
    try:
        # Resolve data path relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "processed" / "pcos_processed.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_pcos_model():
    """Load the trained PCOS model"""
    try:
        model_path = Path(__file__).parent.parent / "models"
        predictor = PCOSPredictor(str(model_path))
        success = predictor.load_model("pcos_model.pkl")
        if success:
            return predictor
        else:
            st.error("Failed to load model")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_dataset_summary(df):
    """Get summary statistics for the dataset"""
    if df is None:
        return {}
    
    summary = {
        'total_records': len(df),
        'pcos_cases': df['pcos_y_n'].sum() if 'pcos_y_n' in df.columns else 0,
        'pcos_percentage': (df['pcos_y_n'].sum() / len(df) * 100) if 'pcos_y_n' in df.columns else 0,
        'avg_age': df['age_yrs'].mean() if 'age_yrs' in df.columns else 0,
        'avg_bmi': df['bmi'].mean() if 'bmi' in df.columns else 0,
    }
    return summary


def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(
        height=600,
        title_font_size=16,
        font=dict(size=10)
    )
    return fig


def create_pcos_distribution_chart(df):
    """Create PCOS distribution chart"""
    if 'pcos_y_n' not in df.columns:
        return None
    
    pcos_counts = df['pcos_y_n'].value_counts()
    labels = ['No PCOS', 'PCOS']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=pcos_counts.values,
            hole=0.4,
            marker_colors=['#5eead4', '#fda4af']
        )
    ])
    
    fig.update_layout(
        title="PCOS Distribution in Dataset",
        title_font_size=16,
        height=400
    )
    return fig


def create_age_bmi_scatter(df):
    """Create Age vs BMI scatter plot"""
    if 'age_yrs' not in df.columns or 'bmi' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='age_yrs',
        y='bmi',
        color='pcos_y_n' if 'pcos_y_n' in df.columns else None,
        color_discrete_map={0: '#5eead4', 1: '#fda4af'},
        title="Age vs BMI Distribution",
        labels={'age_yrs': 'Age (years)', 'bmi': 'BMI', 'pcos_y_n': 'PCOS Status'}
    )
    
    fig.update_layout(
        height=500,
        title_font_size=16
    )
    return fig


def create_hormone_comparison(df):
    """Create hormone level comparison chart"""
    hormone_cols = ['lh_miu_ml', 'fsh_miu_ml', 'amh_ng_ml', 'prl_ng_ml']
    available_cols = [col for col in hormone_cols if col in df.columns]
    
    if not available_cols or 'pcos_y_n' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_cols[:4],
        vertical_spacing=0.1
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, col in enumerate(available_cols[:4]):
        row, col_pos = positions[i]
        
        # Box plot for each hormone by PCOS status
        for pcos_status in [0, 1]:
            data = df[df['pcos_y_n'] == pcos_status][col]
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f"{'PCOS' if pcos_status else 'No PCOS'}",
                    marker_color='#fda4af' if pcos_status else '#5eead4',
                    showlegend=(i == 0)
                ),
                row=row, col=col_pos
            )
    
    fig.update_layout(
        height=600,
        title_text="Hormone Levels Comparison",
        title_font_size=16
    )
    return fig


def create_prediction_gauge(probability):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PCOS Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#fda4af" if probability > 0.5 else "#5eead4"},
            'steps': [
                {'range': [0, 30], 'color': "#f0fdfa"},
                {'range': [30, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#ffe4e9"}
            ],
            'threshold': {
                'line': {'color': "#e11d48", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig


def preprocess_user_input(input_data, df):
    """Preprocess user input to match model expectations"""
    # Get feature names from the dataset (excluding target)
    feature_cols = [col for col in df.columns if col != 'pcos_y_n']
    
    # Create a dataframe with user input
    processed_data = {}
    
    # Map user input to expected feature names
    for col in feature_cols:
        if col in input_data:
            processed_data[col] = input_data[col]
        else:
            # Fill missing values with dataset median
            processed_data[col] = df[col].median() if col in df.columns else 0
    
    return processed_data


def get_health_recommendations(prediction_result):
    """Get personalized health recommendations based on prediction"""
    if prediction_result['prediction'] == 1:  # PCOS positive
        return {
            'diet': [
                "Focus on low glycemic index foods",
                "Include anti-inflammatory foods like leafy greens",
                "Limit processed foods and refined sugars",
                "Consider omega-3 rich foods like fish and nuts"
            ],
            'exercise': [
                "Regular cardio exercises (30 min, 5 days/week)",
                "Strength training 2-3 times per week",
                "Yoga and stress-reduction activities",
                "High-intensity interval training (HIIT)"
            ],
            'lifestyle': [
                "Maintain regular sleep schedule (7-9 hours)",
                "Practice stress management techniques",
                "Monitor weight regularly",
                "Consider working with a healthcare provider"
            ]
        }
    else:  # PCOS negative
        return {
            'diet': [
                "Maintain a balanced, nutritious diet",
                "Include variety of fruits and vegetables",
                "Stay hydrated throughout the day",
                "Practice portion control"
            ],
            'exercise': [
                "Stay active with regular physical activity",
                "Mix cardio and strength training",
                "Find activities you enjoy",
                "Aim for 150 minutes moderate activity per week"
            ],
            'lifestyle': [
                "Maintain healthy sleep habits",
                "Manage stress effectively",
                "Regular health check-ups",
                "Stay informed about women's health"
            ]
        }


# Custom CSS for styling
def load_css():
    """Load custom CSS for the application"""
    return """
    <style>
    /* Import Modern Wellness Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Nunito:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Modern Wellness Design */
    .stApp {
        background: linear-gradient(135deg, #fef5ff 0%, #f0f9ff 30%, #fef3f2 60%, #fef7ed 100%) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        background: transparent;
        font-family: 'Inter', 'Poppins', sans-serif;
        animation: fadeIn 0.8s ease-in;
        max-width: 1400px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar Styling - Soft Pastel Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fef5ff 0%, #f5f3ff 50%, #fce7f3 100%) !important;
        box-shadow: 2px 0 20px rgba(167, 139, 250, 0.08) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #fef5ff 0%, #f5f3ff 50%, #fce7f3 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #6b21a8 !important;
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] h2 {
        color: #6b21a8 !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 4px rgba(107, 33, 168, 0.1);
        font-family: 'Nunito', sans-serif;
    }
    
    [data-testid="stSidebar"] p {
        color: #7c3aed !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #6b21a8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Sidebar Selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #fefbff 100%) !important;
        border: 2px solid #a78bfa !important;
        color: #6b21a8 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(167, 139, 250, 0.15) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: #7c3aed !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sidebar Selectbox Options */
    [data-testid="stSidebar"] [role="option"] {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%) !important;
        color: #6b21a8 !important;
        padding: 0.8rem 1rem !important;
        margin: 0.3rem 0.5rem !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background: linear-gradient(135deg, #e9d5ff 0%, #ddd6fe 100%) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2) !important;
    }
    
    [data-testid="stSidebar"] [aria-selected="true"] {
        background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] .element-container {
        color: #6b21a8 !important;
    }
    
    [data-testid="stSidebar"] strong {
        color: #7c3aed !important;
    }
    
    /* Sidebar Navigation Items Styling */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: white !important;
        border: 2px solid #e9d5ff !important;
        padding: 0.6rem 1rem !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #7c3aed !important;
    }
    
    /* Complete Dropdown Menu Override - Remove Dark Mode */
    [data-testid="stSidebar"] ul[role="listbox"],
    [data-testid="stSidebar"] div[data-baseweb="popover"],
    [data-testid="stSidebar"] div[role="listbox"] {
        background: linear-gradient(135deg, #ffffff 0%, #fef5ff 100%) !important;
        border: 3px solid #e9d5ff !important;
        border-radius: 20px !important;
        padding: 0.8rem !important;
        box-shadow: 0 12px 40px rgba(167, 139, 250, 0.3) !important;
    }
    
    [data-testid="stSidebar"] li[role="option"],
    [data-testid="stSidebar"] div[role="option"] {
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%) !important;
        color: #6b21a8 !important;
        padding: 1rem 1.5rem !important;
        margin: 0.4rem 0 !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 2px solid transparent !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    [data-testid="stSidebar"] li[role="option"]:hover,
    [data-testid="stSidebar"] div[role="option"]:hover {
        background: linear-gradient(135deg, #e9d5ff 0%, #ddd6fe 100%) !important;
        transform: translateX(10px) scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.3) !important;
        border-color: #c4b5fd !important;
        color: #5b21b6 !important;
    }
    
    [data-testid="stSidebar"] li[aria-selected="true"],
    [data-testid="stSidebar"] div[aria-selected="true"] {
        background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.4) !important;
        border-color: #a855f7 !important;
    }
    
    /* Main Header - Modern Wellness Card */
    .main-header {
        background: linear-gradient(135deg, #fae8ff 0%, #e9d5ff 50%, #ddd6fe 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 28px;
        color: #6b21a8;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(167, 139, 250, 0.25);
        border: 3px solid rgba(233, 213, 255, 0.5);
        animation: slideUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(-30px, -30px); }
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #6b21a8 !important;
        text-shadow: 0 4px 20px rgba(107, 33, 168, 0.15);
        font-family: 'Poppins', sans-serif;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.25rem;
        margin: 0;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        color: #7c3aed !important;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards - Modern Wellness Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fefbff 100%);
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(167, 139, 250, 0.15);
        border: 3px solid #f5f3ff;
        margin-bottom: 2rem;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #c084fc 0%, #a855f7 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 20px 50px rgba(167, 139, 250, 0.25);
        border-color: #e9d5ff;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        color: #7c3aed !important;
        margin-bottom: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        background: linear-gradient(135deg, #14b8a6 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Nunito', sans-serif;
    }
    
    /* Page Headers - Warm & Friendly */
    h1, h2, h3 {
        color: #6b21a8 !important;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
    }
    
    /* Text Visibility - Soft Dark */
    p, div, span, label {
        color: #4a5568 !important;
        font-family: 'Poppins', sans-serif;
        line-height: 1.7;
    }
    
    /* Prediction Results - Gentle Colors */
    .prediction-result {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        font-size: 1.4rem;
        font-weight: 600;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.08);
        animation: slideUp 0.8s ease-out;
        font-family: 'Nunito', sans-serif;
    }
    
    .positive-result {
        background: linear-gradient(135deg, #fff5f7 0%, #ffe4e9 100%);
        border: 3px solid #fda4af;
        color: #e11d48 !important;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        border: 3px solid #5eead4;
        color: #0d9488 !important;
    }
    
    /* Health Tips - Lavender Tinted */
    .health-tip {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        padding: 2rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 5px solid #a78bfa;
        box-shadow: 0 4px 16px rgba(167, 139, 250, 0.15);
        border: 2px solid #e9d5ff;
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    .health-tip:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(167, 139, 250, 0.2);
    }
    
    .health-tip h4 {
        color: #7c3aed !important;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.15rem;
        font-family: 'Nunito', sans-serif;
    }
    
    .health-tip p, .health-tip li {
        color: #5b21b6 !important;
        line-height: 1.8;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Buttons - Modern Gradient with Smooth Animation */
    .stButton > button {
        background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 1.1rem 3rem !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.35) !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(168, 85, 247, 0.45) !important;
        background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Input Fields - Modern Dark Gray Style with Blue Focus */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input,
    .stSlider > div > div > div > div {
        background: #1f2937 !important;
        border: 2px solid #374151 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        background: #111827 !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5), 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(-2px) !important;
        outline: none !important;
    }
    
    .stNumberInput > div > div > input:hover,
    .stSelectbox > div > div > select:hover,
    .stTextInput > div > div > input:hover,
    .stDateInput > div > div > input:hover {
        background: #111827 !important;
        border-color: #4b5563 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Dropdown specific styling */
    .stSelectbox > div > div {
        background: #1f2937 !important;
        border-radius: 12px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background: #1f2937 !important;
        border: 2px solid #374151 !important;
        border-radius: 12px !important;
        color: white !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div:hover {
        background: #111827 !important;
        border-color: #4b5563 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Dropdown Menu Options - Dark Style */
    .stSelectbox ul[role="listbox"] {
        background: #1f2937 !important;
        border: 2px solid #374151 !important;
        border-radius: 12px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2) !important;
        padding: 0.5rem !important;
    }
    
    .stSelectbox li[role="option"] {
        background: #374151 !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        margin: 0.3rem 0 !important;
        border-radius: 8px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox li[role="option"]:hover {
        background: #4b5563 !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stSelectbox li[aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Labels - Friendly Purple */
    .stNumberInput label, .stSelectbox label, .stTextInput label, .stDateInput label, .stSlider label {
        color: #6b21a8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        font-family: 'Poppins', sans-serif !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure slider wrapper containers are transparent (remove any pink/gradient capsules) */
    .stSlider > div,
    .stSlider > div > div,
    .stSlider > div > div > div {
        background: transparent !important;
        padding: 0.5rem 0 !important;
        box-shadow: none !important;
        border: 0 !important;
        border-radius: 0 !important;
    }
    
    /* Slider Track - thin, minimal (Dark track) */
    .stSlider [data-baseweb="slider"] {
        height: 2px !important;
        background: #2a2f3a !important;
        border-radius: 999px !important;
    }
    
    /* Slider Track - Filled Part (Solid Red) */
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: #ef4444 !important;
        height: 2px !important;
        border-radius: 999px !important;
    }
    
    /* Slider Thumb - tiny, smooth, minimal (Red) */
    .stSlider [role="slider"] {
        width: 10px !important;
        height: 10px !important;
        background: #ef4444 !important;
        border: none !important; 
        border-radius: 999px !important;
        box-shadow: none !important; 
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }
    
    /* Slider Thumb - Hover State (no glow) */
    .stSlider [role="slider"]:hover { background: #ef4444 !important; }
    
    /* Slider Thumb - Active/Dragging State (no glow) */
    .stSlider [role="slider"]:active { background: #dc2626 !important; }
    
    /* Slider Value Labels */
    .stSlider [data-testid="stTickBar"] {
        color: #6b21a8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    /* Number Input Buttons */
    .stNumberInput button {
        background: #374151 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput button:hover {
        background: #4b5563 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Tabs - Modern Wellness Pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, #fef5ff 0%, #f5f3ff 100%);
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(167, 139, 250, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #ffffff 0%, #fefbff 100%);
        border-radius: 16px;
        color: #7c3aed !important;
        font-weight: 600;
        padding: 1rem 2rem;
        border: 2px solid #e9d5ff;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(167, 139, 250, 0.08);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(167, 139, 250, 0.15);
        border-color: #ddd6fe;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%) !important;
        color: white !important;
        border-color: #a855f7 !important;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.35) !important;
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .css-1r6slb0 {
        background: linear-gradient(135deg, #ffffff 0%, #fefbff 100%) !important;
        border: 2px solid #fce7f3 !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    /* Footer - Soft & Friendly */
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #7c3aed !important;
        border-top: 2px solid #e9d5ff;
        margin-top: 3rem;
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border-radius: 20px;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Success/Error Messages - Soft Pastels */
    .stSuccess {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%) !important;
        border: 2px solid #5eead4 !important;
        color: #0d9488 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #fff5f7 0%, #ffe4e9 100%) !important;
        border: 2px solid #fda4af !important;
        color: #e11d48 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
        border: 2px solid #fbbf24 !important;
        color: #d97706 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border: 2px solid #93c5fd !important;
        color: #1d4ed8 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Subheaders */
    .css-10trblm {
        color: #6b21a8 !important;
        font-weight: 600 !important;
        font-family: 'Nunito', sans-serif !important;
    }
    
    /* Markdown text */
    .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3 {
        color: #6b21a8 !important;
        font-family: 'Nunito', sans-serif !important;
    }
    
    .css-1629p8f p {
        color: #4a5568 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
    """
