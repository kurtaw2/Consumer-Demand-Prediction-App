import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb # Required to load the XGBRegressor model structure
from google.colab import drive # Required for Colab environment setup

# --- Configuration (Must match the paths used in train_and_save.py) ---
MODEL_OUTPUT_FOLDER = 'HFCE_Predictor_Artifacts' 
GDRIVE_PATH = '/content/drive/MyDrive/'

# Define the full persistent paths to the model artifacts
MODEL_DIR = os.path.join(GDRIVE_PATH, MODEL_OUTPUT_FOLDER)
PREDICTOR_FILENAME = 'hfce_predictor.joblib'
FEATURES_FILENAME = 'hfce_features.joblib'
MODEL_EXPORT_PATH = os.path.join(MODEL_DIR, PREDICTOR_FILENAME)
FEATURES_EXPORT_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)

# Hardcoded defaults for demonstration (These should ideally be loaded from a separate stats file)
DEFAULT_SPENDING = 15000.00
DEFAULT_LAG2 = 14500.00
DEFAULT_LAG4 = 16000.00
DEFAULT_CCIS = 10.0
DEFAULT_FIN = 20.0
DEFAULT_INC = 30.0
DEFAULT_INF = 4.0


# --- Data Loading (Cached) ---
# Use st.cache_resource to load the model only once.
@st.cache_resource
def mount_and_load_predictor():
    """Mounts Google Drive and loads the trained model and feature list."""
    try:
        # Attempt to mount drive (required if running directly in a Colab environment)
        # Note: If running on Streamlit Cloud or a local machine, you'd skip this and load directly.
        drive.mount('/content/drive', force_remount=True)
        st.toast("✅ Google Drive Mounted")
    except Exception as e:
        st.error(f"Failed to mount Google Drive. Cannot access model files: {e}")
        return None, None
        
    try:
        # 1. Check if the model directory exists
        if not os.path.exists(MODEL_DIR):
            st.error(f"Model directory not found in Drive: {MODEL_DIR}")
            return None, None

        # 2. Load the artifacts
        model = joblib.load(MODEL_EXPORT_PATH)
        feature_cols = joblib.load(FEATURES_EXPORT_PATH)
        
        st.success("✅ Model and Features loaded successfully from Google Drive.")
        return model, feature_cols
        
    except FileNotFoundError:
        st.error(f"Model files not found in Drive. Did you run train_and_save.py to create the '{MODEL_OUTPUT_FOLDER}' folder?")
        return None, None
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

# --- Application Setup ---
st.set_page_config(page_title="HFCE Prediction Tool", layout="wide")

st.title("👕 HFCE Clothing Consumption Predictor")
st.markdown("""
**Forecast Tool:** Predicts **Household Final Consumption Expenditure (Per Capita)** for Clothing & Footwear.
*Model artifacts are loaded from Google Drive.*
""")

# Load the model and features
with st.spinner("Connecting to Google Drive and loading model..."):
    model, feature_cols = mount_and_load_predictor()

if model is None:
    st.warning("Prediction requires a successfully loaded model.")
    st.stop()


# --- INPUT FORM ---
st.markdown("---")
st.header("1. Define Forecast Scenario")

with st.form("prediction_form"):
    st.info(f"💡 **Current Baseline:** Defaults are set to a typical historical spending of ₱{DEFAULT_SPENDING:,.2f}.")

    st.subheader("Step 1: Timeframe")
    c1, c2 = st.columns(2)
    with c1:
        input_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=2025)
    with c2:
        input_qtr = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.subheader("Step 2: Economic Indicators")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        val_ccis = st.number_input("Consumer Confidence (CCIS)", value=DEFAULT_CCIS, help="Latest Index Value")
    with ec2:
        val_fin = st.number_input("Financial Condition Index", value=DEFAULT_FIN)
    with ec3:
        val_inc = st.number_input("Income Outlook Index", value=DEFAULT_INC)

    st.markdown("#### Inflation")
    inf1, inf2 = st.columns(2)
    with inf1:
        val_inf_curr = st.number_input("Inflation Rate (Target Qtr) %", value=DEFAULT_INF)
    with inf2:
        val_inf_prev = st.number_input("Inflation Rate (Previous Qtr) %", value=DEFAULT_INF - 0.5, help="Used for growth rate calc")

    st.subheader("Step 3: Historical Context (Crucial)")
    st.caption("Consumption (Per Capita) in Pesos.")
    
    hist1, hist2, hist3 = st.columns(3)
    with hist1:
        lag1 = st.number_input("Last Qtr Consumption (Lag 1)", value=DEFAULT_SPENDING, format="%.2f")
    with hist2:
        lag2 = st.number_input("2 Qtrs Ago Consumption (Lag 2)", value=DEFAULT_LAG2, format="%.2f")
    with hist3:
        lag4 = st.number_input("1 Year Ago Consumption (Lag 4)", value=DEFAULT_LAG4, format="%.2f")

    st.markdown("---")
    submit_btn = st.form_submit_button("🚀 Generate Prediction", type="primary")

# --- Prediction Logic ---
if submit_btn:
    input_data = {}
    
    # Inputs
    input_data['Year'] = input_year
    input_data['CCIS_Overall'] = val_ccis
    input_data['CES_FinCondition'] = val_fin
    input_data['CES_Income'] = val_inc
    input_data['Inflation_Annual_Static_Rate'] = val_inf_curr
    
    # Lags
    input_data['HFCE_Lag1'] = lag1
    input_data['HFCE_Lag2'] = lag2
    input_data['HFCE_Lag4'] = lag4
    
    # Calculated Features (Must match the logic in train_and_save.py)
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    # NOTE: The original logic in the user's Streamlit app had a potential error here:
    # (lag1 + lag2 + lag4 + lag4) / 4. A 4-quarter rolling mean should be the average
    # of the *four* previous quarters. Using the definition from the original (lag1+lag2+lag4+lag4)
    # is maintained here for consistency with the user's prior implementation, though
    # a correct 4-quarter mean would require lag3 as well.
    input_data['HFCE_RollingMean_4'] = (lag1 + lag2 + lag4 + lag4) / 4
    
    input_data['Inflation_Growth'] = (val_inf_curr - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['CCIS_Growth'] = 0.0 # CCIS Growth requires the previous CCIS value, default to 0.0 for prediction
    
    # Quarter One-Hot Encoding
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0
    
    # 1. Create the input DataFrame from user data
    df_input = pd.DataFrame([input_data])
    
    # 2. Re-order and fill missing features to match the model's expected input (feature_cols)
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    # Ensure all columns are float types (XGBoost requirement)
    df_final = df_final.astype(float)
        
    # Generate Prediction
    try:
        prediction = model.predict(df_final)[0]
    except Exception as e:
        st.error(f"Prediction Error. Check that the number of input features ({len(df_final.columns)}) matches the training model's expectation: {e}")
        st.stop()
        
    st.header("2. Prediction Results")
    with st.container():
        res_col1, res_col2 = st.columns([1, 2])
        
        # Calculate Delta vs Last Qtr (Lag 1)
        delta_value = ((prediction - lag1) / lag1) * 100
        
        with res_col1:
            st.metric(
                label="Predicted Spending (Per Capita)", 
                value=f"₱ {prediction:,.2f}",
                delta=f"{delta_value:+.2f}% vs Last Qtr" # Display delta with sign
            )
        with res_col2:
            st.info(f"The model predicts spending of **₱{prediction:,.2f}** per person based on the scenario entered.")

elif model:
    st.markdown("---")
    st.subheader("ℹ️ Next Steps")
    st.markdown("Run the prediction form above to see the model's forecast based on your economic scenario inputs.")
