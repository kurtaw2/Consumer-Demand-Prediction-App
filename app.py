import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="HFCE Prediction Tool", layout="wide")

st.title("👕 HFCE Clothing Consumption Predictor")
st.markdown("""
**Forecast Tool:** Enter the economic indicators below to predict the 
**Household Final Consumption Expenditure (Per Capita)** for Clothing & Footwear.
""")

# --- Configuration ---
# Assumes files are in the same directory as app.py
DATA_FOLDER = '.' 
FILES = {
    'hfce': os.path.join(DATA_FOLDER, 'HFCE_data.csv'),
    'pop': os.path.join(DATA_FOLDER, 'Population_data.csv'),
    'inf': os.path.join(DATA_FOLDER, 'Inflation_data.csv'),
    'ces1': os.path.join(DATA_FOLDER, 'CES_Tab_1.csv'),
    'ces3': os.path.join(DATA_FOLDER, 'CES_Tab_3.csv'),
    'ces4': os.path.join(DATA_FOLDER, 'CES_Tab_4.csv'),
}

# --- Helper Functions (Data Loading) ---

def extract_ces_indicator(file_path, indicator_search_term, new_column_name):
    """
    Robustly parses CES files to find the Quarter row and the specific Indicator row.
    """
    try:
        if not os.path.exists(file_path): 
            return pd.DataFrame()
            
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1')  
        
        # 1. Find the row containing Q1, Q2, Q3
        quarter_row_idx = None
        for idx, row in df_raw.iterrows():
            row_text = " ".join(row.astype(str).values)
            if 'Q1' in row_text and 'Q3' in row_text:
                quarter_row_idx = idx
                break
        
        if quarter_row_idx is None: 
            return pd.DataFrame()
        
        year_row_idx = quarter_row_idx - 1
        indicator
