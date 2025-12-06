import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb 

# --- Configuration ---
# DYNAMIC FOLDER DETECTION
found_folders = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('HFCE_Predictor_Artifacts')]
MODEL_OUTPUT_FOLDER = sorted(found_folders)[-1] if found_folders else 'HFCE_Predictor_Artifacts'

MODEL_DIR = os.path.join(os.getcwd(), MODEL_OUTPUT_FOLDER)
PREDICTOR_FILENAME = 'hfce_predictor.joblib'
FEATURES_FILENAME = 'hfce_features.joblib'
DEFAULTS_FILENAME = 'hfce_defaults.joblib'

MODEL_EXPORT_PATH = os.path.join(MODEL_DIR, PREDICTOR_FILENAME)
FEATURES_EXPORT_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)
DEFAULTS_EXPORT_PATH = os.path.join(MODEL_DIR, DEFAULTS_FILENAME)

# Fallback defaults
DEFAULT_VALUES_FALLBACK = {
    'HFCE_Per_Capita': 15000.00,
    'HFCE_Lag1': 15000.00,
    'HFCE_Lag2': 14500.00,
    'HFCE_Lag4': 16000.00,
    'CCIS_Overall': 10.0,
    'CES_FinCondition': 20.0,
    'CES_Income': 30.0,
    'Inflation_Annual_Static_Rate': 4.0,
    'Year': 2025
}

# --- Data Loading ---
@st.cache_resource
def load_predictor():
    feature_cols = None
    defaults = DEFAULT_VALUES_FALLBACK.copy()
    model = None

    if not os.path.exists(MODEL_DIR):
        st.error(f"❌ Error: Model directory '{MODEL_OUTPUT_FOLDER}' not found.")
        return None, None, defaults

    try:
        model = joblib.load(MODEL_EXPORT_PATH)
        feature_cols = joblib.load(FEATURES_EXPORT_PATH)
        if os.path.exists(DEFAULTS_EXPORT_PATH):
            defaults.update(joblib.load(DEFAULTS_EXPORT_PATH))
        return model, feature_cols, defaults
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, defaults

# --- UI Setup ---
st.set_page_config(page_title="Future Textile Demand Forecaster", layout="wide")

# Custom CSS to match the dark theme aesthetics better
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-value { font-size: 36px; font-weight: bold; color: #4CAF50; }
    .metric-delta { font-size: 16px; color: #FF5252; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Future Textile Demand Forecaster")
st.markdown("""
Predict Per-Capita Clothing Spending based on economic indicators. Enter your economic outlook below 
to forecast demand for the next quarter.
""")

# Load Resources
with st.spinner("Initializing forecaster..."):
    model, feature_cols, defaults = load_predictor()

if model is None:
    st.stop()

# --- INPUT SECTION ---
with st.form("prediction_form"):
    
    # Row 1: Economic Indicators
    c1, c2, c3 = st.columns(3)
    
    with c1:
        val_inf = st.number_input("Expected Inflation Rate (%)", value=defaults.get('Inflation_Annual_Static_Rate'), step=0.1)
    
    with c2:
        val_ccis = st.number_input("Expected Consumer Confidence", value=defaults.get('CCIS_Overall'), step=1.0)
    
    with c3:
        # Simplified Quarter Selection
        q_map = {"Q1 (Jan-Mar)": "Q1", "Q2 (Apr-Jun)": "Q2", "Q3 (Jul-Sep)": "Q3", "Q4 (Oct-Dec)": "Q4"}
        q_display = st.selectbox("Target Quarter", list(q_map.keys()))
        input_qtr = q_map[q_display]

    st.markdown("### Historical Inputs")
    
    # Row 2: Historical Data
    h1, h2, h3 = st.columns(3)
    
    with h1:
        lag1 = st.number_input("Prev Quarter Spending (₱)", value=defaults.get('HFCE_Lag1'), format="%.2f")
    
    with h2:
        lag4 = st.number_input("Last Year's Spending (₱)", value=defaults.get('HFCE_Lag4'), format="%.2f")
        
    with h3:
        # Pre-calculate a default rolling mean for display
        default_rolling = (defaults.get('HFCE_Lag1') + defaults.get('HFCE_Lag2') + defaults.get('HFCE_Lag4')*2)/4
        user_rolling = st.number_input("Last 4 Qtr Avg (₱)", value=default_rolling, format="%.2f")

    # --- Hidden / Background Inputs (Required by Model) ---
    # These use defaults since we removed them from the UI to clean it up
    input_year = int(defaults.get('Year') + 1)
    val_fin = defaults.get('CES_FinCondition')
    val_inc = defaults.get('CES_Income')
    val_inf_prev = defaults.get('Inflation_Annual_Static_Rate') # Assume stable inflation for growth calc
    lag2 = defaults.get('HFCE_Lag2') # Hidden Lag 2

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("🚀 Predict Demand", type="primary")

# --- Logic & Output ---
if submit:
    # 1. Prepare Data
    input_data = {}
    input_data['Year'] = input_year
    input_data['CCIS_Overall'] = val_ccis
    input_data['CES_FinCondition'] = val_fin
    input_data['CES_Income'] = val_inc
    input_data['Inflation_Annual_Static_Rate'] = val_inf
    
    input_data['HFCE_Lag1'] = lag1
    input_data['HFCE_Lag2'] = lag2
    input_data['HFCE_Lag4'] = lag4
    
    # Logic: If user manually changes Rolling Mean, we respect that input
    # Otherwise, the model would normally calculate it. Here we inject the user input.
    input_data['HFCE_RollingMean_4'] = user_rolling
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2 # Still needs Lag2, using default
    
    input_data['Inflation_Growth'] = (val_inf - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['CCIS_Growth'] = 0.0

    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0

    # 2. Align & Predict
    df_input = pd.DataFrame([input_data])
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    try:
        prediction = model.predict(df_final.astype(float))[0]
        
        # 3. Custom Output Display (Matching Screenshot)
        st.markdown("---")
        
        r1, r2 = st.columns(2)
        
        with r1:
            st.markdown("Previous Quarter (Input)")
            st.markdown(f"<div class='metric-value'>₱{lag1:,.2f}</div>", unsafe_allow_html=True)
            
        with r2:
            st.markdown("Forecasted Quarter")
            st.markdown(f"<div class='metric-value'>₱{prediction:,.2f}</div>", unsafe_allow_html=True)
            
            # Calculate Percentage Change
            pct_change = ((prediction - lag1) / lag1) * 100
            color = "#FF5252" if pct_change < 0 else "#4CAF50"
            st.markdown(f"<div style='color: {color}; font-weight: bold; background-color: #262730; padding: 4px 8px; border-radius: 4px; display: inline-block;'>{pct_change:+.1f}%</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        direction = "increase" if pct_change > 0 else "decrease"
        st.info(f"Demand is forecasted to **{direction}** by **{abs(pct_change):.1f}%** compared to the previous period.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
