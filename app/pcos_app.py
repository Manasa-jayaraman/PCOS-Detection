
"""
PCOS Health Companion - Professional Multi-Page Streamlit Application

A comprehensive application for PCOS Detection and Health Guidance
"""

import sys
import json
from pathlib import Path

# Add current directory and parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.metrics import accuracy_score

# Page configuration - only set when running this file directly
if __name__ == "__main__":
    st.set_page_config(
        page_title="PCOS Health Companion",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Add parent directory to path for imports
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# --- Startup evaluation: load saved model & dataset and print metrics to terminal ---
try:
    import pickle
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    model_path = Path(__file__).parent.parent / "models" / "pcos_model.pkl"
    data_path = Path(__file__).parent.parent / "data" / "processed" / "pcos_processed.csv"

    if model_path.exists() and data_path.exists():
        try:
            # Load dataset
            df_all = pd.read_csv(data_path)

            # Detect target column
            if 'pcos_y_n' in df_all.columns:
                target_col = 'pcos_y_n'
            else:
                # fallback: look for column names containing 'pcos'
                candidates = [c for c in df_all.columns if 'pcos' in c.lower()]
                target_col = candidates[0] if candidates else None

            if target_col is None:
                print(f"[startup-eval] Could not detect target column in {data_path}")
            else:
                # Prepare X/y
                df_all = df_all.dropna(subset=[target_col])
                y = df_all[target_col].astype(int)
                X = df_all[[c for c in df_all.columns if c != target_col]]

                # Load model
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
                if isinstance(model_obj, dict):
                    model = model_obj.get('model')
                    feature_names = model_obj.get('feature_names')
                else:
                    model = model_obj
                    feature_names = None

                # If the model expects a specific feature order, align X
                if feature_names:
                    available = [c for c in feature_names if c in X.columns]
                    X = X[available]

                # Split for train/test evaluation (80/20 stratified)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                def safe_metrics(model, Xs, ys):
                    y_pred = model.predict(Xs)
                    y_proba = None
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(Xs)[:, 1]
                    except Exception:
                        y_proba = None

                    metrics = {
                        'accuracy': float(accuracy_score(ys, y_pred)),
                        'precision': float(precision_score(ys, y_pred, zero_division=0)),
                        'recall': float(recall_score(ys, y_pred, zero_division=0)),
                        'f1': float(f1_score(ys, y_pred, zero_division=0)),
                    }
                    if y_proba is not None:
                        try:
                            metrics['roc_auc'] = float(roc_auc_score(ys, y_proba))
                        except Exception:
                            metrics['roc_auc'] = None
                    else:
                        metrics['roc_auc'] = None

                    # specificity
                    try:
                        tn, fp, fn, tp = confusion_matrix(ys, y_pred).ravel()
                        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
                    except Exception:
                        metrics['specificity'] = None

                    return metrics, y_pred

                # Compute metrics
                train_metrics, y_train_pred = safe_metrics(model, X_train, y_train)
                test_metrics, y_test_pred = safe_metrics(model, X_test, y_test)

                # Print formatted summary to terminal
                print('\n' + '='*60)
                print('PCOS Saved Model - Startup Evaluation')
                print('='*60)
                print('\nTRAINING SET:')
                print(f"Samples: {len(y_train)}")
                print(f"Accuracy : {train_metrics['accuracy']:.4f}")
                print(f"Precision: {train_metrics['precision']:.4f}")
                print(f"Recall   : {train_metrics['recall']:.4f}")
                print(f"F1 Score : {train_metrics['f1']:.4f}")
                if train_metrics['roc_auc'] is not None:
                    print(f"ROC-AUC  : {train_metrics['roc_auc']:.4f}")
                if train_metrics['specificity'] is not None:
                    print(f"Specificity: {train_metrics['specificity']:.4f}")

                print('\nTEST SET:')
                print(f"Samples: {len(y_test)}")
                print(f"Accuracy : {test_metrics['accuracy']:.4f}")
                print(f"Precision: {test_metrics['precision']:.4f}")
                print(f"Recall   : {test_metrics['recall']:.4f}")
                print(f"F1 Score : {test_metrics['f1']:.4f}")
                if test_metrics['roc_auc'] is not None:
                    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
                if test_metrics['specificity'] is not None:
                    print(f"Specificity: {test_metrics['specificity']:.4f}")

                print('\nConfusion Matrix (Test):')
                try:
                    cm = confusion_matrix(y_test, y_test_pred)
                    print(cm)
                except Exception:
                    pass

                print('\n' + '='*60 + '\n')

        except Exception as e:
            print(f"[startup-eval] Error computing startup metrics: {e}")
    else:
        print(f"[startup-eval] Model or data file not found (model: {model_path.exists()}, data: {data_path.exists()})")
except Exception as _e:
    # Don't break the app if startup evaluation fails
    print(f"[startup-eval] Unexpected error: {_e}")
# --- end startup evaluation ---

# Import utility functions
from utils import (
    load_processed_data, load_pcos_model, get_dataset_summary,
    create_correlation_heatmap, create_pcos_distribution_chart,
    create_age_bmi_scatter, create_hormone_comparison,
    create_prediction_gauge, preprocess_user_input,
    get_health_recommendations, load_css
)

# Database helpers
from db import (
    init_db, create_user, verify_user,
    save_prediction, get_predictions,
    save_daily_log, get_daily_logs,
)

# Global light theme override (neutral with soft green accents)
st.markdown(
    """
    <style>
    .stApp { background: #f8fafc !important; }
    .main .block-container { background: transparent !important; }
    h1, h2, h3 { color: #111827 !important; text-shadow: none !important; }
    p, div, span, label { color: #1f2937 !important; }
    /* Cards */
    .main-header, .metric-card, .health-tip, .prediction-result { background: #ffffff !important; border: 1px solid #e5e7eb !important; box-shadow: none !important; }
    .metric-card::before { background: #10b981 !important; }
    /* Buttons */
    .stButton > button { background: #10b981 !important; border-radius: 10px !important; box-shadow: none !important; color:#ffffff !important; }
    .stButton > button:hover { background: #059669 !important; }
    /* Inputs */
    .stNumberInput input, .stTextInput input, .stDateInput input { background:#ffffff !important; border:1px solid #d1d5db !important; color:#111827 !important; border-radius:10px !important; }
    .stSelectbox [data-baseweb="select"] > div { background:#ffffff !important; border:1px solid #d1d5db !important; border-radius:10px !important; color:#111827 !important; }
    .stSelectbox ul[role="listbox"] { background:#ffffff !important; border:1px solid #e5e7eb !important; }
    /* Sidebar */
    [data-testid="stSidebar"] { background:#ffffff !important; box-shadow: inset -1px 0 0 #e5e7eb !important; }
    [data-testid="stSidebar"] * { color:#374151 !important; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:#f3f4f6 !important; }
    .stTabs [data-baseweb="tab"] { background:#ffffff !important; border:1px solid #e5e7eb !important; color:#334155 !important; }
    .stTabs [aria-selected="true"] { background:#10b981 !important; color:#ffffff !important; border-color:#059669 !important; }
    /* Nav bar overrides */
    .nav-bar { background:#ffffff !important; border: 1px solid #e5e7eb !important; box-shadow: 0 6px 18px rgba(0,0,0,0.04) !important; }
    #top-nav .stButton>button { background:#10b981 !important; }
    #top-nav .stButton>button:hover { filter: brightness(0.95) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Final-final overrides to remove remaining black backgrounds (dropdown menu, steppers, submit area)
st.markdown(
    """
    <style>
    /* BaseWeb Select popover menu (dropdown) */
    div[data-baseweb="popover"] { background: #ffffff !important; border: 1px solid #e5e7eb !important; }
    div[data-baseweb="popover"] ul[role="listbox"],
    div[data-baseweb="popover"] div[role="listbox"] {
        background: #ffffff !important; color: #111827 !important; border: 1px solid #e5e7eb !important;
    }
    div[data-baseweb="popover"] li[role="option"] { background: #ffffff !important; color: #111827 !important; }
    div[data-baseweb="popover"] li[role="option"]:hover { background: #f3f4f6 !important; color: #111827 !important; }

    /* Number input stepper cluster */
    .stNumberInput > div > div { background: #ffffff !important; }
    .stNumberInput button { background: #ffffff !important; color: #111827 !important; border: 1px solid #d1d5db !important; }

    /* Form submit area strip */
    [data-testid="stFormSubmitButton"],
    [data-testid="stFormSubmitButton"] > div { background: transparent !important; }

    /* Ensure disabled submit button stays readable */
    .pp-form .stButton > button:disabled { background: #1e3a8a !important; color: #ffffff !important; opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ultimate overrides: place after all previous CSS to ensure visibility (white controls)
st.markdown(
    """
    <style>
    /* Inputs: force white backgrounds everywhere */
    .stNumberInput input,
    .stTextInput input,
    .stDateInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] input {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }
    /* Number input stepper buttons */
    .stNumberInput button {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    /* Dropdown list panels */
    .stSelectbox ul[role="listbox"],
    .stSelectbox div[role="listbox"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }
    /* Primary buttons: normal and disabled states */
    .stButton > button {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: #f9fafb !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
        filter: none !important;
    }
    .stButton > button:disabled {
        background: #ffffff !important;
        color: #9ca3af !important;
        border: 1px solid #e5e7eb !important;
        opacity: 1 !important;
    }
    /* Keep inputs white on focus */
    .stNumberInput input:focus,
    .stTextInput input:focus,
    .stDateInput input:focus,
    .stSelectbox [data-baseweb="select"] input:focus {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #93c5fd !important;
        box-shadow: 0 0 0 2px rgba(147,197,253,0.35) !important;
    }
    /* Sidebar Logout button: light blue with white text */
    [data-testid="stSidebar"] .stButton > button {
        background: #60a5fa !important; /* light blue */
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
    }
    /* Personal Plan form styles */
    .pp-form .stSelectbox [data-baseweb="select"] > div,
    .pp-form .stNumberInput input,
    .pp-form .stTextInput input {
        color: #111827 !important;
        background: #ffffff !important;
    }
    .pp-form .stSelectbox [data-baseweb="select"]:hover > div {
        color: #111827 !important;
        background: #ffffff !important;
    }
    /* Personal Plan submit button: white text on dark blue */
    .pp-form .stButton > button {
        background: #1e3a8a !important;
        color: #ffffff !important;
        border: 1px solid #1e40af !important;
    }
    .pp-form .stButton > button:disabled {
        background: #1e3a8a !important;
        color: #e5e7eb !important;
        opacity: 1 !important;
    }
    .pp-form .stButton > button:hover {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Final overrides: ensure inputs and primary buttons are white and clearly visible
st.markdown(
    """
    <style>
    /* Inputs: force white backgrounds and dark text */
    .stNumberInput input,
    .stTextInput input,
    .stDateInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] input {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }
    /* Dropdown list */
    .stSelectbox ul[role="listbox"],
    .stSelectbox div[role="listbox"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }
    /* Primary buttons: white background, dark text, subtle border */
    .stButton > button {
        background: #ffffff !important;
        color: #065f46 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        filter: none !important;
        background: #f9fafb !important;
        border-color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
from utils import (
    preprocess_user_input,
    get_health_recommendations, load_css
)

# Database helpers
from db import (
    init_db, create_user, verify_user,
    save_prediction, get_predictions,
    save_daily_log, get_daily_logs,
)

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Global neutral theme override (remove purple tones)
st.markdown(
    """
    <style>
    /* Backgrounds */
    .stApp { background: #f7f7f7 !important; }
    .main .block-container { background: transparent !important; }
    /* Headings and text */
    h1, h2, h3 { color: #0f172a !important; text-shadow: none !important; }
    p, div, span, label { color: #1f2937 !important; }
    /* Cards/sections */
    .main-header, .metric-card, .health-tip, .prediction-result { background: #ffffff !important; border: 1px solid #e5e7eb !important; box-shadow: none !important; }
    .metric-card::before { background: #2563eb !important; }
    /* Buttons */
    .stButton > button { background: #2563eb !important; border-radius: 8px !important; box-shadow: none !important; }
    .stButton > button:hover { background: #1d4ed8 !important; }
    /* Inputs */
    .stNumberInput input, .stTextInput input, .stDateInput input { background:#ffffff !important; border:1px solid #d1d5db !important; color:#111827 !important; border-radius:8px !important; }
    .stSelectbox [data-baseweb="select"] > div { background:#ffffff !important; border:1px solid #d1d5db !important; border-radius:8px !important; color:#111827 !important; }
    .stSelectbox ul[role="listbox"] { background:#ffffff !important; border:1px solid #e5e7eb !important; }
    /* Sliders */
    .stSlider [data-baseweb="slider"] { background:#e5e7eb !important; }
    .stSlider [data-baseweb="slider"] > div:first-child { background:#2563eb !important; }
    .stSlider [role="slider"] { background:#2563eb !important; border:2px solid #ffffff !important; }
    /* Sidebar */
    [data-testid="stSidebar"] { background:#ffffff !important; box-shadow: inset -1px 0 0 #e5e7eb !important; }
    [data-testid="stSidebar"] * { color:#374151 !important; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:#f3f4f6 !important; }
    .stTabs [data-baseweb="tab"] { background:#ffffff !important; border:1px solid #e5e7eb !important; color:#334155 !important; }
    .stTabs [aria-selected="true"] { background:#2563eb !important; color:#ffffff !important; border-color:#1d4ed8 !important; }
    /* Success/Warning/Error/Info */
    .stSuccess, .stError, .stWarning, .stInfo { background:#ffffff !important; border:1px solid #e5e7eb !important; color:#374151 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure DB is initialized
@st.cache_resource
def _init_db_once():
    init_db()
    return True

_ = _init_db_once()

# Initialize session state
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = []
if 'user' not in st.session_state:
    st.session_state.user = None
if 'nav' not in st.session_state:
    st.session_state.nav = 'Dashboard'


def auth_panel():
    """Render login/signup when user is not authenticated."""
    tabs = st.tabs(["Login", "Sign Up"])
    with tabs[0]:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email", placeholder="abc@gmail.com")
        password = st.text_input("Password", type="password", key="login_pwd", placeholder="••••••••")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Login", key="login_btn", use_container_width=True):
            user = verify_user(email, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome back, {user['name'] or user['email']}!")
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tabs[1]:
        st.subheader("Sign Up")
        name = st.text_input("Name", key="signup_name", placeholder="Your full name")
        new_email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
        new_pwd = st.text_input("Password", type="password", key="signup_pwd", placeholder="Create a strong password")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age", min_value=1, max_value=120, value=25, key="signup_age")
        with col_b:
            dob = st.date_input("Date of Birth", key="signup_dob")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Create Account", key="signup_btn", use_container_width=True):
            ok, msg = create_user(new_email, name, new_pwd, int(age), str(dob))
            if ok:
                st.success("Account created! Please login.")
            else:
                st.error(msg or "Failed to create account")


def require_login():
    if not st.session_state.user:
        st.info("Please login or sign up to continue.")
        auth_panel()
        st.stop()

# Global data loading
@st.cache_data
def get_data():
    return load_processed_data()

@st.cache_resource
def get_model():
    return load_pcos_model()

@st.cache_data
def get_model_accuracy():
    """Compute model accuracy once using the loaded dataset and trained model."""
    try:
        df = get_data()
        predictor = get_model()
        if df is None or predictor is None or 'pcos_y_n' not in df.columns:
            return None
        y = df['pcos_y_n']
        X = df.drop(columns=['pcos_y_n'], errors='ignore')

        y_pred = None
        # Try common prediction interfaces
        if hasattr(predictor, 'model') and hasattr(predictor.model, 'predict'):
            y_pred = predictor.model.predict(X)
        elif hasattr(predictor, 'predict'):
            try:
                y_pred = predictor.predict(X)
            except Exception:
                y_pred = None
        if y_pred is None:
            return None
        return float(accuracy_score(y, y_pred))
    except Exception:
        return None


def dashboard_page():
    """Dashboard page with summary and project info"""
    st.markdown("""
    <div class="main-header">
        <h1>✨ Welcome to Your PCOS Health Companion</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem; color: white;">
            💜 Your personal wellness assistant for understanding and managing PCOS with care
        </p>
    </div>
    """, unsafe_allow_html=True)
    # Scoped slider styling: sleek 3px track, modern handles, clear spacing
    st.markdown(
        """
        <style>
        .slider-label {font-weight: 700; color: #6b21a8; margin: 8px 0 6px 0;}
        .slider-block {margin: 10px 0 22px 0;}
        /* Slim rail and track */
        .stSlider .rc-slider-rail {height: 3px !important; background: #e9d5ff !important; border-radius: 999px !important;}
        .stSlider .rc-slider-track {height: 3px !important; background: linear-gradient(90deg,#a78bfa,#f0abfc) !important; border-radius: 999px !important;}
        /* Handle */
        .stSlider .rc-slider-handle {width: 14px !important; height: 14px !important; margin-top: -6px !important; border: 2px solid #ffffff !important; background: #8b5cf6 !important; box-shadow: 0 6px 14px rgba(139,92,246,.35) !important;}
        .stSlider .rc-slider-handle:hover {box-shadow: 0 8px 18px rgba(139,92,246,.45) !important;}
        /* Dots/steps subtle */
        .stSlider .rc-slider-dot {width: 6px !important; height: 6px !important; background: #e9d5ff !important; border: 0 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Load data and get summary
    df = get_data()
    if df is not None:
        summary = get_dataset_summary(df)
        
        # Summary cards
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📋 Total Records</h3>
                <h2>{:,}</h2>
            </div>
            """.format(summary['total_records']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 PCOS Cases</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(summary['pcos_percentage']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📏 Average BMI</h3>
                <h2>{:.1f}</h2>
            </div>
            """.format(summary['avg_bmi']), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Average Age</h3>
                <h2>{:.1f} years</h2>
            </div>
            """.format(summary['avg_age']), unsafe_allow_html=True)
    
    # About sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What We're Here For")
        st.markdown("""
        <div class="health-tip">
        We're here to support you on your wellness journey! 🌸 This friendly companion uses smart technology 
        to help you understand PCOS better and provides caring, personalized guidance every step of the way.
        
        **What Makes Us Special:**
        - 🔮 Gentle AI-powered health insights
        - 📊 Easy-to-understand data visualizations
        - 🥗 Personalized wellness recommendations
        - 📈 Track your progress with love & care
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Model Accuracy card in the right-side unused space
        acc = get_model_accuracy()
        if acc is not None:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>🎯 Model Accuracy</h3>
                    <h2>{acc:.1f}%</h2>
                </div>
                """.format(acc=acc*100),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>🎯 Model Accuracy</h3>
                    <h2>—</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

    


def analytics_page():
    """Analytics page with interactive visualizations"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 2.5rem; border-radius: 24px; margin-bottom: 2rem; box-shadow: 0 8px 24px rgba(167, 139, 250, 0.15); border: 2px solid #e9d5ff;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.8rem; font-weight: 700; color: #6b21a8;">
            📊 Discover Your Data Story
        </h1>
        <p style="font-size: 1.1rem; margin-bottom: 0; color: #7c3aed;">
            ✨ Explore beautiful insights and patterns in PCOS data - knowledge is power!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = get_data()
    if df is None:
        st.error("Unable to load dataset. Please check the data files.")
        return
    
    # Show all records (removed slider filters)
    filtered_df = df
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 1rem 1.5rem; border-radius: 16px; margin: 1.5rem 0; border-left: 5px solid #14b8a6; box-shadow: 0 4px 12px rgba(20, 184, 166, 0.15);">
        <p style="margin: 0; color: #0d9488; font-weight: 500; font-size: 1rem;">📈 Showing all <strong>{len(filtered_df)}</strong> records</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations with section header
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="color: #6b21a8; font-size: 1.5rem; margin-bottom: 0.5rem;">📊 Visual Insights</h3>
        <p style="color: #7c3aed; font-size: 0.95rem;">Interactive charts to help you understand the data better</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCOS Distribution
        fig_dist = create_pcos_distribution_chart(filtered_df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Age vs BMI Scatter
        fig_scatter = create_age_bmi_scatter(filtered_df)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="color: #6b21a8; font-size: 1.5rem; margin-bottom: 0.5rem;">🔥 Feature Relationships</h3>
        <p style="color: #7c3aed; font-size: 0.95rem;">See how different health factors connect with each other</p>
    </div>
    """, unsafe_allow_html=True)
    fig_corr = create_correlation_heatmap(filtered_df)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Hormone Comparison
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="color: #6b21a8; font-size: 1.5rem; margin-bottom: 0.5rem;">🧪 Hormone Level Patterns</h3>
        <p style="color: #7c3aed; font-size: 0.95rem;">Understanding hormone variations across different groups</p>
    </div>
    """, unsafe_allow_html=True)
    fig_hormones = create_hormone_comparison(filtered_df)
    if fig_hormones:
        st.plotly_chart(fig_hormones, use_container_width=True)


def prediction_page():
    """Prediction page with input form and results"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 50%, #e0f2fe 100%); padding: 2.5rem; border-radius: 24px; margin-bottom: 2rem; box-shadow: 0 8px 24px rgba(94, 234, 212, 0.2); border: 2px solid #5eead4;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.8rem; font-weight: 700; color: #0d9488;">
            🔮 Your Personal Health Assessment
        </h1>
        <p style="font-size: 1.1rem; margin-bottom: 0; color: #0891b2;">
            💫 Let's understand your health together - share your information and we'll provide caring insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = get_model()
    if predictor is None:
        st.error("Model not available. Please check the model files.")
        return
    
    df = get_data()
    if df is None:
        st.error("Unable to load dataset for preprocessing.")
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 4px 16px rgba(167, 139, 250, 0.15); border: 2px solid #e9d5ff;">
        <h3 style="color: #7c3aed; margin-bottom: 0.8rem; font-size: 1.3rem;">📝 Share Your Health Information</h3>
        <p style="color: #5b21b6; margin-bottom: 0; font-size: 0.95rem;">💙 Your privacy matters! All information stays secure and is only used for your assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff5f7 0%, #ffe4e9 100%); padding: 1rem; border-radius: 16px; margin-bottom: 1rem; border-left: 4px solid #fda4af;">
            <h4 style="color: #e11d48; margin: 0; font-size: 1.1rem;">📋 Basic Information</h4>
        </div>
        """, unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=15, max_value=50, value=25, help="Your current age")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=65.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=140.0, max_value=200.0, value=165.0, step=0.1)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        st.metric("Calculated BMI", f"{bmi:.1f}")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 16px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #fbbf24;">
            <h4 style="color: #d97706; margin: 0; font-size: 1.1rem;">🩸 Hormone Levels</h4>
        </div>
        """, unsafe_allow_html=True)
        lh = st.number_input("LH (mIU/mL)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        fsh = st.number_input("FSH (mIU/mL)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        amh = st.number_input("AMH (ng/mL)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 1rem; border-radius: 16px; margin-bottom: 1rem; border-left: 4px solid #5eead4;">
            <h4 style="color: #0d9488; margin: 0; font-size: 1.1rem;">🔬 Clinical Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        cycle_regular = st.selectbox("Menstrual Cycle", ["Regular", "Irregular"])
        cycle_length = st.number_input("Cycle Length (days)", min_value=20, max_value=45, value=28)
        
        weight_gain = st.selectbox("Recent Weight Gain", ["No", "Yes"])
        hair_growth = st.selectbox("Excessive Hair Growth", ["No", "Yes"])
        skin_darkening = st.selectbox("Skin Darkening", ["No", "Yes"])
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1rem; border-radius: 16px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #60a5fa;">
            <h4 style="color: #1d4ed8; margin: 0; font-size: 1.1rem;">🍎 Lifestyle</h4>
        </div>
        """, unsafe_allow_html=True)
        fast_food = st.selectbox("Regular Fast Food", ["No", "Yes"])
        exercise = st.selectbox("Regular Exercise", ["No", "Yes"])
        
        # Additional parameters with defaults
        pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=60, max_value=100, value=72)
        bp_systolic = st.number_input("Systolic BP", min_value=90, max_value=160, value=120)
    
    # Prediction button
    if st.button("🔍 Analyze PCOS Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            try:
                # Prepare input data
                input_data = {
                    'age_yrs': age,
                    'weight_kg': weight,
                    'height_cm': height,
                    'bmi': bmi,
                    'lh_miu_ml': lh,
                    'fsh_miu_ml': fsh,
                    'amh_ng_ml': amh,
                    'cycle_r_i': 2 if cycle_regular == "Regular" else 1,
                    'cycle_length_days': cycle_length,
                    'weight_gain_y_n': 1 if weight_gain == "Yes" else 0,
                    'hair_growth_y_n': 1 if hair_growth == "Yes" else 0,
                    'skin_darkening_y_n': 1 if skin_darkening == "Yes" else 0,
                    'fast_food_y_n': 1 if fast_food == "Yes" else 0,
                    'reg_exercise_y_n': 1 if exercise == "Yes" else 0,
                    'pulse_rate_bpm': pulse_rate,
                    'bp_systolic_mmhg': bp_systolic
                }
                
                # Preprocess input
                processed_input = preprocess_user_input(input_data, df)
                
                # Make prediction
                result = predictor.predict_single(processed_input)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                if result['prediction'] == 1:
                    probability = result['probability']['pcos'] if result['probability'] else 0.5
                    st.markdown(f"""
                    <div class="prediction-result positive-result">
                        💝 <strong>We're Here to Support You</strong><br>
                        Our analysis suggests elevated PCOS indicators<br>
                        Confidence: {result['confidence']:.1%}<br>
                        <small>Remember: Early awareness empowers better care! 🌸</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    probability = result['probability']['no_pcos'] if result['probability'] else 0.5
                    st.markdown(f"""
                    <div class="prediction-result negative-result">
                        ✨ <strong>Looking Good!</strong><br>
                        Lower PCOS risk indicators detected<br>
                        Confidence: {result['confidence']:.1%}<br>
                        <small>Keep up the great work with your wellness! 🌟</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge
                col1, col2 = st.columns([1, 1])
                with col1:
                    if result['probability']:
                        pcos_prob = result['probability']['pcos']
                        fig_gauge = create_prediction_gauge(pcos_prob)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Recommendations
                    recommendations = get_health_recommendations(result)
                    st.subheader("💡 Your Personalized Next Steps")
                    
                    if result['prediction'] == 1:
                        st.warning("💙 **We recommend:** Connect with a healthcare provider who can support you with personalized care")
                    
                    st.info("**Gentle Suggestions to Start:**\n" + 
                           "\n".join([f"• {rec}" for rec in recommendations['lifestyle'][:3]]))
                # Persist prediction for logged in user
                if st.session_state.user:
                    try:
                        save_prediction(
                            user_id=st.session_state.user['id'],
                            input_json=json.dumps(input_data),
                            prediction=int(result['prediction']),
                            p_pcos=(result['probability']['pcos'] if result['probability'] else None),
                            p_no_pcos=(result['probability']['no_pcos'] if result['probability'] else None),
                            confidence=float(result['confidence']) if result.get('confidence') is not None else None,
                        )
                    except Exception as e:
                        st.warning(f"Could not save prediction: {e}")

                # Show recent history
                if st.session_state.user:
                    st.markdown("### Your Previous Health Predictions")
                    rows = get_predictions(st.session_state.user['id'], limit=5)
                    if rows:
                        hist_df = pd.DataFrame(rows)
                        st.dataframe(hist_df[["created_at", "prediction", "p_pcos", "confidence"]])
                    else:
                        st.info("No saved predictions yet.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # Reset button
    if st.button("🔄 Reset Form"):
        st.rerun()


def health_guidance_page():
    """Health guidance page with tips and progress tracker"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff5f7 0%, #ffe4e9 50%, #fef3c7 100%); padding: 2.5rem; border-radius: 24px; margin-bottom: 2rem; box-shadow: 0 8px 24px rgba(253, 164, 175, 0.2); border: 2px solid #fda4af;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.8rem; font-weight: 700; color: #e11d48;">
            🥗 Your Wellness Toolkit
        </h1>
        <p style="font-size: 1.1rem; margin-bottom: 0; color: #f43f5e;">
            💚 Caring guidance and tools to support your journey to better health - one step at a time!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Health tips sections
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 4px 16px rgba(167, 139, 250, 0.15); border: 2px solid #e9d5ff;">
        <h3 style="color: #7c3aed; margin-bottom: 0.5rem; font-size: 1.3rem;">💡 Your Personal Wellness Guide</h3>
        <p style="color: #5b21b6; margin-bottom: 0; font-size: 0.95rem; font-style: italic;">Small changes can make a big difference! Let's explore together. 🌸</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different guidance categories
    tab1, tab2, tab3, tab4 = st.tabs(["🍎 Diet", "🏋️‍♀️ Exercise", "💤 Lifestyle", "📈 Progress Tracker"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h3 style="color: #7c3aed; font-size: 1.5rem; margin-bottom: 0.5rem;">🥗 Nutritional Guidance for PCOS</h3>
            <p style="color: #5b21b6; font-size: 0.95rem;">Nourish your body with foods that support your wellness</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="health-tip">
            <h4>✅ Recommended Foods</h4>
            • <strong>Low GI foods:</strong> Quinoa, brown rice, oats<br>
            • <strong>Lean proteins:</strong> Fish, chicken, legumes<br>
            • <strong>Anti-inflammatory:</strong> Leafy greens, berries<br>
            • <strong>Healthy fats:</strong> Avocado, nuts, olive oil<br>
            • <strong>Fiber-rich:</strong> Vegetables, whole grains
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="health-tip">
            <h4>❌ Foods to Limit</h4>
            • <strong>Processed foods:</strong> Fast food, packaged snacks<br>
            • <strong>Refined sugars:</strong> Candy, sodas, desserts<br>
            • <strong>Simple carbs:</strong> White bread, pasta<br>
            • <strong>Trans fats:</strong> Fried foods, margarine<br>
            • <strong>Excess dairy:</strong> Limit high-fat dairy products
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h3 style="color: #7c3aed; font-size: 1.5rem; margin-bottom: 0.5rem;">🏋️‍♀️ Exercise Recommendations</h3>
            <p style="color: #5b21b6; font-size: 0.95rem;">Move your body in ways that feel good and build strength</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="health-tip">
            <h4>💪 Strength Training</h4>
            • <strong>Frequency:</strong> 2-3 times per week<br>
            • <strong>Focus:</strong> Compound movements<br>
            • <strong>Benefits:</strong> Improves insulin sensitivity<br>
            • <strong>Examples:</strong> Squats, deadlifts, push-ups
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="health-tip">
            <h4>🏃 Cardio & HIIT</h4>
            • <strong>Moderate cardio:</strong> 150 min/week<br>
            • <strong>HIIT sessions:</strong> 2-3 times/week<br>
            • <strong>Low impact:</strong> Swimming, cycling<br>
            • <strong>Yoga:</strong> Stress reduction & flexibility
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h3 style="color: #7c3aed; font-size: 1.5rem; margin-bottom: 0.5rem;">🌙 Lifestyle Management</h3>
            <p style="color: #5b21b6; font-size: 0.95rem;">Create daily habits that nurture your mind and body</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="health-tip">
            <h4>😴 Sleep Hygiene</h4>
            • <strong>Duration:</strong> 7-9 hours nightly<br>
            • <strong>Consistency:</strong> Same bedtime daily<br>
            • <strong>Environment:</strong> Cool, dark, quiet<br>
            • <strong>Avoid:</strong> Screens before bed
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="health-tip">
            <h4>🧘‍♀️ Stress Management</h4>
            • <strong>Meditation:</strong> 10-15 min daily<br>
            • <strong>Deep breathing:</strong> 4-7-8 technique<br>
            • <strong>Hobbies:</strong> Reading, music, art<br>
            • <strong>Social support:</strong> Friends, family, groups
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h3 style="color: #7c3aed; font-size: 1.5rem; margin-bottom: 0.5rem;">📊 Personal Progress Tracker</h3>
            <p style="color: #5b21b6; font-size: 0.95rem;">Track your journey and celebrate your progress!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress input form
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 1.2rem; border-radius: 16px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #14b8a6;">
            <h4 style="color: #0d9488; margin: 0; font-size: 1.1rem;">📝 Daily Health Log</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Date", datetime.date.today())
            weight_today = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=65.0, step=0.1)
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=8.0, step=0.5)
        
        with col2:
            exercise_minutes = st.number_input("Exercise (minutes)", min_value=0, max_value=300, value=30)
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            mood = st.selectbox("Mood", ["Poor", "Fair", "Good", "Excellent"])
        
        with col3:
            water_glasses = st.number_input("Water (glasses)", min_value=0, max_value=20, value=8)
            symptoms = st.multiselect("Symptoms Today", 
                                    ["Fatigue", "Mood swings", "Cravings", "Bloating", "Acne", "Hair issues"])
        
        if st.button("💾 Save Today's Data"):
            entry = {
                'date': date,
                'weight': weight_today,
                'sleep': sleep_hours,
                'exercise': exercise_minutes,
                'stress': stress_level,
                'mood': mood,
                'water': water_glasses,
                'symptoms': symptoms
            }
            st.session_state.progress_data.append(entry)
            # Save to DB if logged in
            if st.session_state.user:
                try:
                    save_daily_log(
                        user_id=st.session_state.user['id'],
                        date=str(date),
                        weight=weight_today,
                        sleep=sleep_hours,
                        exercise_minutes=exercise_minutes,
                        stress=stress_level,
                        mood=mood,
                        water=water_glasses,
                        symptoms=", ".join(symptoms) if symptoms else "",
                    )
                except Exception as e:
                    st.warning(f"Could not save daily log: {e}")
            st.success("🎉 Wonderful! Your data has been saved. You're doing great by tracking your progress!")
        
        # Display progress charts
        # Load persisted logs if logged in
        persisted_logs = []
        if st.session_state.user:
            try:
                persisted_logs = get_daily_logs(st.session_state.user['id'], limit=365)
            except Exception:
                persisted_logs = []

        combined_logs = st.session_state.progress_data.copy()
        if persisted_logs:
            combined_logs = persisted_logs

        if combined_logs:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 1.2rem; border-radius: 16px; margin: 2rem 0 1rem 0; border-left: 4px solid #a78bfa;">
                <h4 style="color: #7c3aed; margin: 0; font-size: 1.2rem;">📈 Your Progress Trends</h4>
                <p style="color: #5b21b6; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Watch your wellness journey unfold!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Convert to DataFrame
            progress_df = pd.DataFrame(combined_logs)
            progress_df['date'] = pd.to_datetime(progress_df['date'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Weight trend
                fig_weight = px.line(progress_df, x='date', y='weight', 
                                   title='Weight Trend', markers=True)
                fig_weight.update_traces(line_color='#14b8a6', marker=dict(size=8))
                st.plotly_chart(fig_weight, use_container_width=True)
            
            with col2:
                # Sleep trend
                fig_sleep = px.line(progress_df, x='date', y='sleep', 
                                  title='Sleep Hours Trend', markers=True)
                fig_sleep.update_traces(line_color='#a78bfa', marker=dict(size=8))
                st.plotly_chart(fig_sleep, use_container_width=True)
        
        else:
            st.info("\ud83d\udcab Ready to start your wellness journey? Log your first entry above and watch your progress bloom!")

def render_personal_plan_section(embed: bool = False):
    """Render Personal Plan inputs and generated cards. If embed=True, render as a section."""
    st.markdown('<a name="personal-plan"></a>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#fff; padding: 1rem 1.25rem; border-radius: 12px; margin: 0.25rem 0 1rem 0; border: 1px solid #e5e7eb;">
            <h2 style="margin:0;">📝 Personal Plan</h2>
            <p style="margin:.2rem 0 0 0; color:#475569;">Enter your details and generate a simple, practical plan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='pp-form'>", unsafe_allow_html=True)
    with st.form("pp_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=13, max_value=80, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
            menstrual = st.selectbox("Menstrual Cycle Pattern", ["Regular", "Irregular", "Unknown"], index=0)
            restrictions = st.selectbox("Food Restrictions", ["None", "Lactose-free", "Gluten-free", "Nut-free", "Low-sugar"]) 
            sleeping_hours = st.number_input("Sleeping Hours (per day)", min_value=0.0, max_value=14.0, value=8.0, step=0.5)
        with c2:
            height = st.number_input("Height (cm)", min_value=135.0, max_value=210.0, value=165.0)
            activity_label = st.selectbox(
                "Activity Level",
                [
                    "Sedentary (little or no exercise)",
                    "Lightly Active (1–3 days/week)",
                    "Moderately Active (3–5 days/week)",
                    "Very Active (6–7 days/week)",
                    "Extra Active (hard exercise/physical job)",
                ],
                index=2,
                help="Choose the description that best matches your weekly activity.",
            )
            diet = st.selectbox("Dietary Preference", ["No preference", "Vegetarian", "Vegan", "Pescatarian", "Non-vegetarian"]) 
            goal = st.selectbox("Primary Goal", ["Maintain", "Weight Loss", "Weight Gain", "Improve Energy", "Symptom Relief"], index=1)
            mood = st.selectbox("Mood", ["Happy", "Normal", "Tired", "Stressed"], index=1)

        submitted = st.form_submit_button("Generate My Plan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Calculations
        bmi = weight / ((height/100) ** 2)
        bmi_cat = "Underweight" if bmi < 18.5 else ("Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese"))
        act_map = {
            "Sedentary (little or no exercise)": 1.2,
            "Lightly Active (1–3 days/week)": 1.375,
            "Moderately Active (3–5 days/week)": 1.55,
            "Very Active (6–7 days/week)": 1.725,
            "Extra Active (hard exercise/physical job)": 1.9,
        }
        act_factor = act_map[activity_label]
        # Mifflin-St Jeor for women (approx, no sex field available)
        bmr = 10*weight + 6.25*height - 5*age - 161
        tdee = max(1200, round(bmr * act_factor))
        if goal == "Weight Loss":
            calories = max(1100, tdee - 300)
        elif goal == "Weight Gain":
            calories = tdee + 300
        else:
            calories = tdee
        # Macros 30/40/30
        protein_g = round(calories * 0.30 / 4)
        carbs_g = round(calories * 0.40 / 4)
        fats_g = round(calories * 0.30 / 9)

        # Diet plan lines
        veg = diet in ["Vegetarian", "Vegan"]
        protein_src = "Tofu/Paneer/Lentils" if veg else "Eggs/Chicken/Fish"
        snack = "Nuts + Fruit" if restrictions != "Nut-free" else "Roasted chickpeas + Fruit"
        carb = "Quinoa/Brown rice/Oats"
        diet_lines = [
            f"Calories ≈ {calories} kcal | Protein {protein_g}g • Carbs {carbs_g}g • Fats {fats_g}g",
            "Breakfast: " + ("Overnight oats + berries + chia" if veg else "Veg omelette + multigrain toast"),
            f"Lunch: {carb} + mixed veggies + {protein_src}",
            f"Snack: {snack}",
            "Dinner: Veg stir-fry + salad + protein",
            "Guideline: High fiber, low GI, hydrate well, limit refined sugar",
        ]

        # Weekly exercise schedule
        base = [
            ("Mon", "Walk 30–40 min"),
            ("Tue", "Strength (full body) 30 min"),
            ("Wed", "Yoga/Stretch 25 min"),
            ("Thu", "Intervals walk/cycle 30 min"),
            ("Fri", "Strength (core + legs) 30 min"),
            ("Sat", "Yoga/Pilates 30 min"),
            ("Sun", "Light walk or rest 20 min"),
        ]
        if bmi >= 30:
            base[0] = ("Mon", "Low‑impact walk 25–30 min")
            base[3] = ("Thu", "Intervals low impact 20–25 min")
        intensity_map = {
            "Sedentary (little or no exercise)": "Low",
            "Lightly Active (1–3 days/week)": "Low‑Moderate",
            "Moderately Active (3–5 days/week)": "Moderate",
            "Very Active (6–7 days/week)": "Moderate‑High",
            "Extra Active (hard exercise/physical job)": "High",
        }
        intensity = intensity_map[activity_label]

        # Persist today's weight, sleep, and mood for tracking
        try:
            today = datetime.date.today()
            # Append to in-session tracker
            st.session_state.progress_data.append({
                'date': str(today),
                'weight': float(weight),
                'sleep': float(sleeping_hours),
                'exercise': 0,
                'stress': 'NA',
                'mood': mood,
                'water': 0,
                'symptoms': ''
            })
            # Save to DB if logged in
            if st.session_state.user:
                try:
                    save_daily_log(
                        user_id=st.session_state.user['id'],
                        date=str(today),
                        weight=float(weight),
                        sleep=float(sleeping_hours),
                        exercise_minutes=0,
                        stress='NA',
                        mood=mood,
                        water=0,
                        symptoms="",
                    )
                except Exception as e:
                    st.warning(f"Could not persist daily log: {e}")
        except Exception:
            pass

        # Results layout
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
                <div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px;">
                    <h3 style="margin:0 0 4px 0;">🍽️ Diet Plan</h3>
                    <p style="margin:0 0 8px 0; color:#475569;">BMI: {bmi:.1f} ({bmi_cat}) · Preference: {diet}</p>
                </div>
                """.format(bmi=bmi, bmi_cat=bmi_cat, diet=diet),
                unsafe_allow_html=True,
            )
            for line in diet_lines:
                st.write(f"• {line}")
        with right:
            st.markdown(
                """
                <div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px;">
                    <h3 style="margin:0 0 4px 0;">💪 Exercise Routine</h3>
                    <p style="margin:0 0 8px 0; color:#475569;">Weekly schedule · Intensity: {intensity}</p>
                </div>
                """.format(intensity=intensity),
                unsafe_allow_html=True,
            )
            for day, desc in base:
                st.write(f"• {day}: {desc}")

        # Guidance snapshots
        st.markdown("---")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.markdown("**Diet Tips**")
            st.write("• Prefer low‑GI carbs, add leafy greens")
            st.write("• Include omega‑3 sources")
        with gc2:
            st.markdown("**Exercise Tips**")
            st.write("• 150 min moderate cardio/week")
            st.write("• 2–3 strength sessions/week")
        with gc3:
            st.markdown("**Lifestyle**")
            st.write("• Sleep 7–9 hrs, reduce stress")
            st.write("• Hydrate 8–10 glasses/day")

        # Download/Save
        plan_text = (
            f"Personal Plan\nBMI: {bmi:.1f} ({bmi_cat})\nCalories: {calories} kcal\n"
            f"Protein {protein_g}g, Carbs {carbs_g}g, Fats {fats_g}g\n\n"
            + "\n".join(diet_lines)
            + "\n\nExercise:\n" + "\n".join([f"{d}: {t}" for d,t in base])
        )
        st.download_button("Download Plan (TXT)", plan_text, file_name="personal_plan.txt")
        if 'saved_plans' not in st.session_state:
            st.session_state.saved_plans = []
        if st.button("Save Plan"):
            st.session_state.saved_plans.append({
                'age': age, 'weight': weight, 'height': height, 'activity': activity_label,
                'diet': diet, 'restrictions': restrictions, 'goal': goal, 'bmi': round(bmi,1),
                'calories': calories, 'protein_g': protein_g, 'carbs_g': carbs_g, 'fats_g': fats_g,
                'diet_lines': diet_lines, 'exercise': base
            })
            st.success("Plan saved in this session.")

def personal_plan_page():
    """Standalone Personal Plan page reusing the section renderer."""
    render_personal_plan_section(embed=False)

def progress_tracker_page():
    """Progress Tracker page: show weight and sleep trends from saved daily logs."""
    st.markdown("""
    <div style="background:#fff; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid #e5e7eb;">
        <h2 style="margin:0;">📈 Progress Tracker</h2>
        <p style="margin:.25rem 0 0 0; color:#475569;">Your weight and sleep trends over time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load persisted logs if logged in
    persisted_logs = []
    if st.session_state.user:
        try:
            persisted_logs = get_daily_logs(st.session_state.user['id'], limit=365)
        except Exception:
            persisted_logs = []

    combined_logs = st.session_state.progress_data.copy()
    if persisted_logs:
        combined_logs = persisted_logs

    if not combined_logs:
        st.info("No progress data yet. Add entries from the Personal Plan page.")
        return

    df_logs = pd.DataFrame(combined_logs)
    if 'date' in df_logs:
        df_logs['date'] = pd.to_datetime(df_logs['date'])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Weight Tracker")
        if 'weight' in df_logs:
            fig_w = px.line(df_logs.sort_values('date'), x='date', y='weight', title='Weight Over Time', markers=True)
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("No weight data available.")
    with c2:
        st.subheader("Sleep Tracker")
        if 'sleep' in df_logs:
            fig_s = px.line(df_logs.sort_values('date'), x='date', y='sleep', title='Sleep Hours Over Time', markers=True)
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("No sleep data available.")

def main():
    """Main application with sidebar navigation"""
    # Require login before entering main experience
    if not st.session_state.user:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 2rem; border-radius: 20px; margin-bottom: 1.5rem; border: 2px solid #e9d5ff;">
            <h2 style="color:#6b21a8; margin:0;">🔐 Welcome — Please Login or Sign Up</h2>
            <p style="color:#7c3aed; margin:0.5rem 0 0 0;">Create an account to save your predictions and health logs.</p>
        </div>
        """, unsafe_allow_html=True)
        auth_panel()
        return

    # Simple top navigation only (brand, five nav buttons, profile)
    st.markdown(
        """
        <style>
        .nav-bar {background: linear-gradient(90deg,#faf5ff,#ffffff); padding: 10px 16px; border-radius: 16px; border: 1px solid #e9d5ff; box-shadow: 0 6px 18px rgba(167,139,250,0.16); margin-bottom: 12px;}
        .brand {font-weight: 800; color: #6b21a8; font-size: 18px;}
        .profile-pill {width: 36px; height: 36px; border-radius: 9999px; background: linear-gradient(135deg,#a78bfa,#f0abfc); color:#fff; display:flex; align-items:center; justify-content:center; border:1px solid #e9d5ff; box-shadow: 0 6px 14px rgba(167,139,250,0.35);}
        /* Scope button styling to top nav row only */
        #top-nav .stButton>button { 
            width: 200px; min-width: 200px; height: 48px; 
            white-space: nowrap !important; word-break: keep-all !important; overflow: hidden; text-overflow: clip; 
            display:flex; align-items:center; justify-content:center; line-height: 1; padding: 8px 12px;
            border-radius: 16px; border: 0; 
            background: linear-gradient(135deg,#a78bfa,#c084fc); color: #ffffff; 
            font-weight: 800; letter-spacing: .03px; font-size: 13px;
            box-shadow: 0 10px 20px rgba(167,139,250,0.35);
            transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
        }
        #top-nav .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(167,139,250,0.45); filter: brightness(1.02); }
        #top-nav .stButton>button:active { transform: translateY(0); filter: brightness(0.98); }
        /* Active underline bar */
        .active-bar {height: 4px; border-radius: 999px; background: linear-gradient(90deg,#a78bfa,#f0abfc); box-shadow: 0 6px 12px rgba(167,139,250,0.35); margin-top: 6px;}
        </style>
        <div class="nav-bar"></div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation and theme
    st.sidebar.markdown("### PCOS Companion")
    nav_choice = st.sidebar.radio(
        "",
        ["Dashboard", "Analytics", "Prediction", "Personal Plan", "Progress Tracker", "History"],
        index=["Dashboard", "Analytics", "Prediction", "Personal Plan", "Progress Tracker", "History"].index(
            st.session_state.nav if st.session_state.nav in ["Dashboard", "Analytics", "Prediction", "Personal Plan", "Progress Tracker", "History"] else "Dashboard"
        )
    )
    st.session_state.nav = nav_choice

    # Remove theme dropdown per request; keep simple divider for spacing
    st.sidebar.markdown("---")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()
    
    # Page routing
    page = st.session_state.nav
    if page == "Dashboard":
        dashboard_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Personal Plan":
        personal_plan_page()
    elif page == "Progress Tracker":
        progress_tracker_page()
    elif page == "History":
        st.subheader("Your History")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Recent Predictions")
            rows = get_predictions(st.session_state.user['id'], limit=50)
            if rows:
                dfp = pd.DataFrame(rows)
                st.dataframe(dfp[["created_at", "prediction", "p_pcos", "confidence"]])
            else:
                st.info("No predictions saved yet.")
        with col2:
            st.markdown("### Daily Logs")
            logs = get_daily_logs(st.session_state.user['id'], limit=365)
            if logs:
                dfl = pd.DataFrame(logs)
                st.dataframe(dfl[["date", "weight", "sleep", "exercise_minutes", "mood", "stress", "water"]])
            else:
                st.info("No daily logs saved yet.")
    
    # Footer (removed per user request)


if __name__ == "__main__":
    main()
