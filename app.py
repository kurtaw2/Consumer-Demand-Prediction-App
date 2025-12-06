import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb # Required to load the XGBRegressor model structure

# --- Configuration ---
# This matches the folder name defined in train_and_save.py
MODEL_OUTPUT_FOLDER = 'HFCE_Predictor_Artifacts' 

# Define the local paths to the model artifacts
MODEL_DIR = os.path.join(os.getcwd(), MODEL_OUTPUT_FOLDER)
PREDICTOR_FILENAME = 'hfce_predictor.joblib'
FEATURES_FILENAME = 'hfce_features.joblib'
DEFAULTS_FILENAME = 'hfce_defaults.joblib'

MODEL_EXPORT_PATH = os.path.join(MODEL_DIR, PREDICTOR_FILENAME)
FEATURES_EXPORT_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)
DEFAULTS_EXPORT_PATH = os.path.join(MODEL_DIR, DEFAULTS_FILENAME)

# Fallback defaults in case the defaults file is missing
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

# --- Data Loading (Cached) ---
@st.cache_resource
def load_predictor():
    """Loads the trained model, feature list, and dynamic defaults from the local repository."""
    
    model = None
    feature_cols = None
    defaults = DEFAULT_VALUES_FALLBACK.copy()

    # 1. Check if directory exists
    if not os.path.exists(MODEL_DIR):
        st.error(f"❌ Error: Model directory not found at: {MODEL_DIR}")
        st.warning(f"Please run 'train_and_save.py' first to generate the '{MODEL_OUTPUT_FOLDER}' folder.")
        return None, None, defaults

    try:
        # 2. Load the artifacts
        model = joblib.load(MODEL_EXPORT_PATH)
        feature_cols = joblib.load(FEATURES_EXPORT_PATH)

        # 3. Try to load dynamic defaults
        if os.path.exists(DEFAULTS_EXPORT_PATH):
            dynamic_defaults = joblib.load(DEFAULTS_EXPORT_PATH)
            defaults.update(dynamic_defaults)
        
        return model, feature_cols, defaults
        
    except FileNotFoundError:
        st.error(f"❌ Critical files missing inside '{MODEL_OUTPUT_FOLDER}'. Ensure .joblib files exist.")
        return None, None, defaults
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None, defaults

# --- Application UI ---
st.set_page_config(page_title="HFCE Prediction Tool", layout="wide")

st.title("👕 HFCE Clothing Consumption Predictor")
st.markdown("""
**Forecast Tool:** Predicts **Household Final Consumption Expenditure (Per Capita)** for Clothing & Footwear
based on economic indicators and historical consumption patterns.
""")

# Load Model
with st.spinner("Loading model from local artifacts..."):
    model, feature_cols, dynamic_defaults = load_predictor()

if model is None:
    st.stop() # Stop execution if model failed to load

# Extract Defaults for UI
def_spending = dynamic_defaults.get('HFCE_Per_Capita')
def_lag1 = dynamic_defaults.get('HFCE_Lag1')
def_lag2 = dynamic_defaults.get('HFCE_Lag2')
def_lag4 = dynamic_defaults.get('HFCE_Lag4')
def_year = dynamic_defaults.get('Year')

# --- Input Form ---
st.markdown("---")
with st.form("prediction_form"):
    st.info(f"💡 **Baseline:** Defaults are populated using the most recent data (Year {def_year}).")

    st.subheader("1. Timeframe")
    c1, c2 = st.columns(2)
    with c1:
        input_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=int(def_year + 1))
    with c2:
        input_qtr = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.subheader("2. Economic Indicators")
    ec1, ec2, ec3 = st.columns(3)
    val_ccis = ec1.number_input("Consumer Confidence (CCIS)", value=dynamic_defaults.get('CCIS_Overall'))
    val_fin = ec2.number_input("Financial Condition Index", value=dynamic_defaults.get('CES_FinCondition'))
    val_inc = ec3.number_input("Income Outlook Index", value=dynamic_defaults.get('CES_Income'))

    st.subheader("3. Inflation Metrics")
    i1, i2 = st.columns(2)
    val_inf_curr = i1.number_input("Inflation Rate (Target Qtr) %", value=dynamic_defaults.get('Inflation_Annual_Static_Rate'))
    val_inf_prev = i2.number_input("Inflation Rate (Previous Qtr) %", value=dynamic_defaults.get('Inflation_Annual_Static_Rate'), help="Required to calculate inflation growth rate.")

    st.subheader("4. Historical Consumption (Pesos)")
    st.caption("Enter the per capita spending for previous quarters to establish the trend.")
    h1, h2, h3 = st.columns(3)
    lag1 = h1.number_input("Last Qtr (Lag 1)", value=def_lag1, format="%.2f")
    lag2 = h2.number_input("2 Qtrs Ago (Lag 2)", value=def_lag2, format="%.2f")
    lag4 = h3.number_input("1 Year Ago (Lag 4)", value=def_lag4, format="%.2f")

    submit_btn = st.form_submit_button("🚀 Generate Prediction", type="primary")

# --- Prediction Logic ---
if submit_btn:
    # 1. Prepare Raw Inputs
    input_data = {}
    
    # Direct mappings
    input_data['Year'] = input_year
    input_data['CCIS_Overall'] = val_ccis
    input_data['CES_FinCondition'] = val_fin
    input_data['CES_Income'] = val_inc
    input_data['Inflation_Annual_Static_Rate'] = val_inf_curr
    input_data['HFCE_Lag1'] = lag1
    input_data['HFCE_Lag2'] = lag2
    input_data['HFCE_Lag4'] = lag4
    
    # Feature Engineering (Must mimic train_and_save.py logic)
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    # Approx for 4-quarter rolling mean (Lag1+Lag2+Lag4+Lag4)/4 as we lack Lag3 input
    input_data['HFCE_RollingMean_4'] = (lag1 + lag2 + lag4 + lag4) / 4 
    
    # Growth Rates
    input_data['Inflation_Growth'] = (val_inf_curr - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['CCIS_Growth'] = 0.0 # Defaulting to 0 as we don't ask for previous CCIS
    
    # One-Hot Encoding for Quarter
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0
    
    # 2. Align with Model Features
    # Create DF from input
    df_input = pd.DataFrame([input_data])
    
    # Create empty DF with correct feature columns (loaded from joblib)
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # Map input values to the correct columns
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    # Ensure float types
    df_final = df_final.astype(float)
        
    # 3. Predict
    try:
        prediction = model.predict(df_final)[0]
        
        # Display Results
        st.markdown("### 📊 Prediction Results")
        res_col1, res_col2 = st.columns([1, 2])
        
        delta_value = ((prediction - lag1) / lag1) * 100
        
        with res_col1:
            st.metric(
                label="Predicted Spending (Per Capita)", 
                value=f"₱ {prediction:,.2f}",
                delta=f"{delta_value:+.2f}% vs Last Qtr"
            )
        with res_col2:
            st.success(f"Forecast: **₱{prediction:,.2f}**")
            st.info("This prediction uses the specific economic conditions and consumption trends you provided.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
