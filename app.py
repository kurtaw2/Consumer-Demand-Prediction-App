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
        indicator_row_idx = None
        
        # 2. Find the data row
        if indicator_search_term == 'ROW_0_fallback':
             # Look for first numeric row after quarters
             for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 20, len(df_raw))):
                 try:
                     val_str = str(df_raw.iloc[idx, 2]).replace(',', '').strip()
                     if val_str and val_str.lower() != 'nan':
                         float(val_str)
                         indicator_row_idx = idx
                         break
                 except: continue
        
        if indicator_row_idx is None: 
            return pd.DataFrame()

        # 3. Extract
        quarter_row = df_raw.iloc[quarter_row_idx]
        year_row = df_raw.iloc[year_row_idx]
        data_row = df_raw.iloc[indicator_row_idx]

        dates = []
        values = []
        current_year = None

        for col_idx in range(1, len(df_raw.columns)):
            val_year = year_row[col_idx]
            if pd.notna(val_year):
                try:
                    s = str(val_year).strip().replace(',', '').replace('.0', '')
                    if s.isdigit() and len(s) == 4: current_year = int(s)
                except: pass
            
            val_qtr = quarter_row[col_idx]
            val_data = data_row[col_idx]

            if pd.notna(val_qtr) and pd.notna(val_data) and current_year is not None:
                qtr_str = str(val_qtr).strip()
                if qtr_str in ['Q1', 'Q2', 'Q3', 'Q4']:
                    dates.append((current_year, qtr_str))
                    values.append(val_data)

        df_cleaned = pd.DataFrame(dates, columns=['Year', 'Quarter'])
        df_cleaned[new_column_name] = pd.to_numeric(pd.Series(values).astype(str).str.replace(',', ''), errors='coerce')
        df_cleaned.dropna(inplace=True)
        return df_cleaned
    except: 
        return pd.DataFrame()

@st.cache_data
def train_model():
    """
    Loads data, cleans it, performs feature engineering, and trains the model.
    Returns the trained model and the list of feature columns expected.
    """
    # 1. Load Files
    try:
        hfce_df = pd.read_csv(FILES['hfce'], header=8, encoding='latin1')
        pop_df = pd.read_csv(FILES['pop'], encoding='latin1')
        inf_df = pd.read_csv(FILES['inf'], encoding='latin1')
    except Exception as e:
        st.error(f"Error loading core files: {e}")
        return None, None

    # 2. Clean HFCE
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    
    # Map Years
    year_map = {}
    current_year = None
    for col_idx, col_value in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_value):
            s = str(col_value).strip()
            if s.isdigit() and len(s)==4: current_year = int(s)
        year_map[col_idx + 1] = current_year
    
    # Map Quarters
    years = []
    quarters = []
    vals = []
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for i, row in hfce_target.iterrows():
        col_idx = hfce_df.columns.get_loc(row['Year'])
        y = year_map.get(col_idx)
        q = quarter_names[(col_idx - 1) % 4]
        
        if y is not None:
            years.append(y)
            quarters.append(q)
            vals.append(row['Value'])

    clean_hfce = pd.DataFrame({'Year': years, 'Quarter': quarters, 'Value': vals})
    clean_hfce['HFCE_Clothing_Footwear'] = pd.to_numeric(clean_hfce['Value'].astype(str).str.replace(',', ''), errors='coerce') * 1_000_000
    clean_hfce.dropna(inplace=True)

    # 3. Clean Population
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Interpolated Quarterly Estimate'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter_Index'] = pop_df.groupby('Year').cumcount()
    pop_df['Quarter'] = pop_df['Quarter_Index'].apply(lambda x: f'Q{x%4+1}')
    
    # 4. Clean Inflation
    inf_df.columns = ['Year', 'Quarter_Int', 'Inflation_Annual_Static_Rate']
    inf_df['Quarter'] = inf_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inf_df['Year'] = pd.to_numeric(inf_df['Year'], errors='coerce')
    
    # 5. Load CES Indicators
    df_ccis = extract_ces_indicator(FILES['ces1'], 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(FILES['ces3'], 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(FILES['ces4'], 'ROW_0_fallback', 'CES_Income')

    # 6. Merge All
    data = pd.merge(clean_hfce[['Year','Quarter','HFCE_Clothing_Footwear']], pop_df[['Year','Quarter','Quarterly_Population']], on=['Year','Quarter'], how='inner')
    
    for d in [df_ccis, df_fin, df_inc]:
        if not d.empty: data = pd.merge(data, d, on=['Year','Quarter'], how='left')
    
    data = pd.merge(data, inf_df[['Year','Quarter','Inflation_Annual_Static_Rate']], on=['Year','Quarter'], how='left')

    # 7. Feature Engineering
    data = data.sort_values(by=['Year', 'Quarter']).fillna(method='ffill').fillna(0)
    data = data[data['Quarterly_Population'] > 0]
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    # Lags & Rolling
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data['HFCE_RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data['HFCE_RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    data['Inflation_Growth'] = data['Inflation_Annual_Static_Rate'].pct_change()
    data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
    
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.dropna(inplace=True) # Drop rows with NaNs from lags
    
    # 8. ML Setup
    y = data['HFCE_Per_Capita']
    X = data.drop(columns=['HFCE_Per_Capita', 'HFCE_Clothing_Footwear', 'Quarterly_Population', 'Value'], errors='ignore')
    if 'Unnamed: 0' in X.columns: X.drop(columns=['Unnamed: 0'], inplace=True)
    
    # One Hot Encoding
    X = pd.get_dummies(X, columns=['Quarter'], drop_first=True)
    
    # Ensure Numeric
    for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Train
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# --- Main Application Logic ---

# 1. Train/Load Model (Hidden in background)
with st.spinner("Initializing Model & Loading Data from Repo..."):
    model, feature_cols = train_model()

if model is None:
    st.error("⚠️ CRITICAL ERROR: Could not load data. Please ensure 'HFCE_data.csv', 'Population_data.csv', etc. are in the repository folder.")
    st.stop()

# 2. Main Input Form (Center Page)
st.markdown("---")
st.header("1. Define Forecast Scenario")

with st.form("prediction_form"):
    st.subheader("Step 1: Timeframe")
    c1, c2 = st.columns(2)
    with c1:
        input_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=2025)
    with c2:
        input_qtr = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.subheader("Step 2: Economic Indicators")
    st.caption("Enter your projections for the target quarter.")
    
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        val_ccis = st.number_input("Consumer Confidence (CCIS)", value=-10.0)
    with ec2:
        val_fin = st.number_input("Financial Condition Index", value=-5.0)
    with ec3:
        val_inc = st.number_input("Income Outlook Index", value=5.0)

    st.markdown("#### Inflation")
    inf1, inf2 = st.columns(2)
    with inf1:
        val_inf_curr = st.number_input("Inflation Rate (Target Qtr) %", value=4.5)
    with inf2:
        val_inf_prev = st.number_input("Inflation Rate (Previous Qtr) %", value=4.2, help="Required to calculate the rate of change.")

    st.subheader("Step 3: Historical Context")
    st.caption("Enter the actual consumption values from the past to anchor the prediction.")
    
    hist1, hist2, hist3 = st.columns(3)
    with hist1:
        lag1 = st.number_input("Last Qtr Consumption (Lag 1)", value=1500.0)
    with hist2:
        lag2 = st.number_input("2 Qtrs Ago Consumption (Lag 2)", value=1450.0)
    with hist3:
        lag4 = st.number_input("1 Year Ago Consumption (Lag 4)", value=1400.0)

    # Centered Submit Button
    st.markdown("---")
    submit_btn = st.form_submit_button("🚀 Generate Prediction", type="primary")

# 3. Prediction Logic & Results
if submit_btn:
    # Construct the input row exactly like the training data
    input_data = {}
    
    # Basic Features
    input_data['Year'] = input_year
    input_data['CCIS_Overall'] = val_ccis
    input_data['CES_FinCondition'] = val_fin
    input_data['CES_Income'] = val_inc
    input_data['Inflation_Annual_Static_Rate'] = val_inf_curr
    
    # Lags
    input_data['HFCE_Lag1'] = lag1
    input_data['HFCE_Lag2'] = lag2
    input_data['HFCE_Lag4'] = lag4
    
    # Calculated Features (Derived from inputs)
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    input_data['HFCE_RollingMean_4'] = (lag1 + lag2 + lag4 + lag4) / 4 # Approx
    
    # Growth Rates
    pct_change_inf = (val_inf_curr - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['Inflation_Growth'] = pct_change_inf
    input_data['CCIS_Growth'] = 0.0 # Simplification for manual input

    # One-Hot Encoding for Quarter
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Initialize df_final with 0s to ensure float dtype immediately
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # Fill in values
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    # CRITICAL FIX: Ensure all data is float/numeric before passing to XGBoost
    df_final = df_final.astype(float)
            
    # PREDICT
    prediction = model.predict(df_final)[0]
    
    # 4. Display Output
    st.header("2. Prediction Results")
    
    # Create a nice visual container
    with st.container():
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(
                label="Predicted Spending (Per Capita)", 
                value=f"₱ {prediction:,.2f}",
                delta=f"{((prediction - lag1)/lag1)*100:.2f}% vs Last Qtr"
            )
        
        with res_col2:
            st.info(f"""
            **Forecast Details:**
            For **{input_year} {input_qtr}**, with an inflation rate of **{val_inf_curr}%** and Consumer Confidence of **{val_ccis}**, the model predicts spending will be **₱{prediction:,.2f}**.
            """)

elif model:
    # 5. "About" Section (Default view when no prediction is active)
    st.markdown("---")
    st.subheader("ℹ️ About this Project")
    
    st.markdown("""
    This application leverages machine learning to forecast **Household Final Consumption Expenditure (HFCE)** specifically for the **Clothing and Footwear** sector in the Philippines. 
    
    By analyzing historical relationships between economic indicators and consumer behavior, the model estimates discretionary spending patterns.
    """)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("**📊 Data Sources:**")
        st.markdown("""
        * **Spending Data:** PSA (Philippine Statistics Authority)
        * **Consumer Sentiment:** BSP (Bangko Sentral ng Pilipinas) Consumer Expectations Survey
        * **Inflation & Population:** PSA
        """)
        
    with col_info2:
        st.markdown("**🧠 Methodology:**")
        st.markdown("""
        The model uses **XGBoost (Extreme Gradient Boosting)**, a robust machine learning algorithm, to identify non-linear correlations between inflation, financial outlook, and consumption.
        """)
