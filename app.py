import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
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

# Custom CSS
st.markdown("""
<style>
    .metric-value { font-size: 36px; font-weight: bold; color: #4CAF50; }
    .metric-sub { font-size: 14px; color: #888; margin-top: -10px; }
    .highlight-box { background-color: #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Future Textile Demand Forecaster")
st.markdown("""
Predict Per-Capita Clothing Spending based on economic indicators. 
**Auto-Calibration enabled** to correct unit mismatches.
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

    # --- Hidden Inputs ---
    input_year = int(defaults.get('Year') + 1)
    val_fin = defaults.get('CES_FinCondition')
    val_inc = defaults.get('CES_Income')
    val_inf_prev = defaults.get('Inflation_Annual_Static_Rate') 
    lag2 = defaults.get('HFCE_Lag2') 

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("🚀 Predict Demand", type="primary")

# --- Logic & Output ---
if submit:
    # 1. Prepare Data Dictionary
    input_data = {}
    input_data['Year'] = input_year
    input_data['CCIS_Overall'] = val_ccis
    input_data['CES_FinCondition'] = val_fin
    input_data['CES_Income'] = val_inc
    input_data['Inflation_Annual_Static_Rate'] = val_inf
    
    input_data['HFCE_Lag1'] = lag1
    input_data['HFCE_Lag2'] = lag2
    input_data['HFCE_Lag4'] = lag4
    
    input_data['HFCE_RollingMean_4'] = user_rolling
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    
    input_data['Inflation_Growth'] = (val_inf - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['CCIS_Growth'] = 0.0

    # Explicitly handle all Quarter possibilities to ensure features are active
    # Note: 'Quarter_Q1' is typically the dropped reference column in training, so we don't set it explicitly
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0

    # 2. Align Data with Model Features
    df_input = pd.DataFrame([input_data])
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    try:
        raw_prediction = model.predict(df_final.astype(float))[0]
        
        # --- AUTO-CALIBRATION FIX ---
        # The model might predict in different units (e.g. Millions vs Pesos).
        # We assume the prediction should be roughly similar magnitude to the Lag1 Input.
        scaling_factor = 1.0
        calibrated_msg = ""
        
        if raw_prediction > 0 and lag1 > 0:
            ratio = lag1 / raw_prediction
            # If off by more than 10x, snap to nearest power of 10
            if ratio > 10 or ratio < 0.1:
                power = round(math.log10(ratio))
                scaling_factor = 10 ** power
                calibrated_msg = f"(Auto-scaled by x{int(scaling_factor)} to match input units)"

        final_prediction = raw_prediction * scaling_factor
        
        # 3. Output Display
        st.markdown("---")
        
        r1, r2, r3 = st.columns(3)
        
        with r1:
            st.markdown("Previous Quarter (Input)")
            st.markdown(f"<div class='metric-value'>₱{lag1:,.2f}</div>", unsafe_allow_html=True)
            
        with r2:
            st.markdown("Forecasted Quarter")
            st.markdown(f"<div class='metric-value'>₱{final_prediction:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>{calibrated_msg}</div>", unsafe_allow_html=True)
            
        with r3:
            # Calculate Percentage Change
            pct_change = ((final_prediction - lag1) / lag1) * 100
            color = "#FF5252" if pct_change < 0 else "#4CAF50"
            st.markdown("Change")
            st.markdown(f"<div style='color: {color}; font-size: 36px; font-weight: bold;'>{pct_change:+.1f}%</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- SEASONALITY CHECKER (To prove quarters work) ---
        with st.expander("🛠️ Advanced Analysis & Debugging"):
            st.write("### Seasonal Sensitivity Check")
            st.write("Does changing the quarter affect the prediction? (Holding other values constant)")
            
            # Create a test batch for all 4 quarters
            test_rows = []
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            for q in quarters:
                row = df_final.copy()
                row['Quarter_Q2'] = 1 if q == 'Q2' else 0
                row['Quarter_Q3'] = 1 if q == 'Q3' else 0
                row['Quarter_Q4'] = 1 if q == 'Q4' else 0
                test_rows.append(row)
            
            test_df = pd.concat(test_rows, ignore_index=True)
            test_preds = model.predict(test_df.astype(float)) * scaling_factor
            
            sensitivity_df = pd.DataFrame({
                "Quarter": quarters, 
                "Forecast (₱)": test_preds,
                "Diff from Avg": test_preds - test_preds.mean()
            })
            st.dataframe(sensitivity_df.style.format({"Forecast (₱)": "{:,.2f}", "Diff from Avg": "{:+,.2f}"}))
            
            if sensitivity_df['Forecast (₱)'].nunique() == 1:
                st.warning("⚠️ The model is not reacting to Quarter changes. This usually means 'Lagged Consumption' is overwhelmingly the dominant feature, washing out seasonal effects.")
            else:
                st.success(f"✅ Seasonality Detected: The model adjusts the forecast by up to ₱{sensitivity_df['Diff from Avg'].abs().max():,.2f} based on the quarter.")

            st.write("### Raw Model Input")
            st.dataframe(df_final)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
