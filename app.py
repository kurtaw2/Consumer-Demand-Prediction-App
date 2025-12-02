import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="HFCE Prediction Tool", layout="wide")

st.title("👕 HFCE Clothing Consumption Predictor")
st.markdown("""
**Forecast Tool:** Predicts **Household Final Consumption Expenditure (Per Capita)** for Clothing & Footwear.
*Note: Inputs are auto-filled with the latest available data from your files to ensure correct scaling.*
""")

# --- Configuration ---
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
    try:
        if not os.path.exists(file_path): return pd.DataFrame()
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1')  
        
        quarter_row_idx = None
        for idx, row in df_raw.iterrows():
            if 'Q1' in " ".join(row.astype(str).values) and 'Q3' in " ".join(row.astype(str).values):
                quarter_row_idx = idx
                break
        if quarter_row_idx is None: return pd.DataFrame()
        
        year_row_idx = quarter_row_idx - 1
        indicator_row_idx = None
        
        if indicator_search_term == 'ROW_0_fallback':
             for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 20, len(df_raw))):
                 try:
                     val_str = str(df_raw.iloc[idx, 2]).replace(',', '').strip()
                     if val_str and val_str.lower() != 'nan':
                         float(val_str)
                         indicator_row_idx = idx
                         break
                 except: continue
        
        if indicator_row_idx is None: return pd.DataFrame()

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
                if str(val_qtr).strip() in ['Q1', 'Q2', 'Q3', 'Q4']:
                    dates.append((current_year, str(val_qtr).strip()))
                    values.append(val_data)

        df_cleaned = pd.DataFrame(dates, columns=['Year', 'Quarter'])
        df_cleaned[new_column_name] = pd.to_numeric(pd.Series(values).astype(str).str.replace(',', ''), errors='coerce')
        df_cleaned.dropna(inplace=True)
        return df_cleaned
    except: return pd.DataFrame()

@st.cache_data
def train_model():
    # 1. Load Files
    try:
        hfce_df = pd.read_csv(FILES['hfce'], header=8, encoding='latin1')
        pop_df = pd.read_csv(FILES['pop'], encoding='latin1')
        inf_df = pd.read_csv(FILES['inf'], encoding='latin1')
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

    # 2. Clean HFCE
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    
    year_map = {}
    current_year = None
    for col_idx, col_value in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_value):
            s = str(col_value).strip()
            if s.isdigit() and len(s)==4: current_year = int(s)
        year_map[col_idx + 1] = current_year
    
    # Extract Year/Quarter
    rows = []
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
    for i, row in hfce_target.iterrows():
        col_idx = hfce_df.columns.get_loc(row['Year'])
        y = year_map.get(col_idx)
        q = quarter_names[(col_idx - 1) % 4]
        if y is not None:
            rows.append({'Year': y, 'Quarter': q, 'Value': row['Value']})

    clean_hfce = pd.DataFrame(rows)
    # Scale: Assuming HFCE is in Millions, multiply by 1M to get actual Pesos
    clean_hfce['HFCE_Clothing_Footwear'] = pd.to_numeric(clean_hfce['Value'].astype(str).str.replace(',', ''), errors='coerce') * 1_000_000
    clean_hfce.dropna(inplace=True)

    # 3. Clean Population
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Interpolated Quarterly Estimate'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter'] = pop_df.groupby('Year').cumcount().apply(lambda x: f'Q{x%4+1}')
    
    # 4. Clean Inflation
    inf_df.columns = ['Year', 'Quarter_Int', 'Inflation_Annual_Static_Rate']
    inf_df['Quarter'] = inf_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inf_df['Year'] = pd.to_numeric(inf_df['Year'], errors='coerce')
    
    # 5. CES
    df_ccis = extract_ces_indicator(FILES['ces1'], 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(FILES['ces3'], 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(FILES['ces4'], 'ROW_0_fallback', 'CES_Income')

    # 6. Merge
    data = pd.merge(clean_hfce[['Year','Quarter','HFCE_Clothing_Footwear']], pop_df[['Year','Quarter','Quarterly_Population']], on=['Year','Quarter'], how='inner')
    for d in [df_ccis, df_fin, df_inc]:
        if not d.empty: data = pd.merge(data, d, on=['Year','Quarter'], how='left')
    data = pd.merge(data, inf_df[['Year','Quarter','Inflation_Annual_Static_Rate']], on=['Year','Quarter'], how='left')

    # 7. Engineering
    data = data.sort_values(by=['Year', 'Quarter']).fillna(method='ffill').fillna(0)
    data = data[data['Quarterly_Population'] > 0]
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data['HFCE_RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data['HFCE_RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    data['Inflation_Growth'] = data['Inflation_Annual_Static_Rate'].pct_change()
    data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
    
    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # We keep the LAST row (even if it has NaNs in future targets) to use as default inputs
    last_known_data = data.iloc[-1].copy()
    
    data.dropna(inplace=True)
    
    # 8. ML
    y = data['HFCE_Per_Capita']
    X = data.drop(columns=['HFCE_Per_Capita', 'HFCE_Clothing_Footwear', 'Quarterly_Population', 'Value'], errors='ignore')
    if 'Unnamed: 0' in X.columns: X.drop(columns=['Unnamed: 0'], inplace=True)
    
    X = pd.get_dummies(X, columns=['Quarter'], drop_first=True)
    for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Reduced max_depth to prevent overfitting on small data
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist(), last_known_data

# --- Main Logic ---

with st.spinner("Initializing Model & Loading Data..."):
    model, feature_cols, last_data = train_model()

if model is None:
    st.error("⚠️ Error loading data. Please check CSV files.")
    st.stop()

# --- INPUT FORM ---
st.markdown("---")
st.header("1. Define Forecast Scenario")

# Use LAST KNOWN DATA as defaults to prevent scale errors
def_year = int(last_data['Year'])
def_ccis = float(last_data.get('CCIS_Overall', 0))
def_fin = float(last_data.get('CES_FinCondition', 0))
def_inc = float(last_data.get('CES_Income', 0))
def_inf = float(last_data.get('Inflation_Annual_Static_Rate', 4.0))
# Lags: The most recent 'Per Capita' value is the Lag1 for the prediction
def_lag1 = float(last_data['HFCE_Per_Capita']) 
def_lag2 = float(last_data['HFCE_Lag1']) # The previous lag1 is now lag2
def_lag4 = float(last_data['HFCE_Lag4']) # Rough approximation for UI defaults

with st.form("prediction_form"):
    st.info(f"💡 **System Check:** The average spending per person in the training data is **₱{def_lag1:,.2f}**. Input values have been auto-set to this scale.")

    st.subheader("Step 1: Timeframe")
    c1, c2 = st.columns(2)
    with c1:
        input_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=def_year + 1)
    with c2:
        input_qtr = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.subheader("Step 2: Economic Indicators")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        val_ccis = st.number_input("Consumer Confidence (CCIS)", value=def_ccis)
    with ec2:
        val_fin = st.number_input("Financial Condition Index", value=def_fin)
    with ec3:
        val_inc = st.number_input("Income Outlook Index", value=def_inc)

    st.markdown("#### Inflation")
    inf1, inf2 = st.columns(2)
    with inf1:
        val_inf_curr = st.number_input("Inflation Rate (Target Qtr) %", value=def_inf)
    with inf2:
        val_inf_prev = st.number_input("Inflation Rate (Previous Qtr) %", value=def_inf, help="Used for growth rate calc")

    st.subheader("Step 3: Historical Context (Crucial)")
    st.caption("These define the 'Trend'. Defaults are set to the most recent known data.")
    
    hist1, hist2, hist3 = st.columns(3)
    with hist1:
        lag1 = st.number_input("Last Qtr Consumption (Lag 1)", value=def_lag1, format="%.2f")
    with hist2:
        lag2 = st.number_input("2 Qtrs Ago Consumption (Lag 2)", value=def_lag2, format="%.2f")
    with hist3:
        lag4 = st.number_input("1 Year Ago Consumption (Lag 4)", value=def_lag4, format="%.2f")

    st.markdown("---")
    submit_btn = st.form_submit_button("🚀 Generate Prediction", type="primary")

# 3. Prediction Logic
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
    
    # Calc
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    input_data['HFCE_RollingMean_4'] = (lag1 + lag2 + lag4 + lag4) / 4 
    input_data['Inflation_Growth'] = (val_inf_curr - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['CCIS_Growth'] = 0.0 

    # Quarter One-Hot
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0
    
    # Build DataFrame
    df_input = pd.DataFrame([input_data])
    df_final = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
            
    # FORCE FLOAT to fix XGBoost error
    df_final = df_final.astype(float)
            
    prediction = model.predict(df_final)[0]
    
    st.header("2. Prediction Results")
    with st.container():
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                label="Predicted Spending (Per Capita)", 
                value=f"₱ {prediction:,.2f}",
                delta=f"{((prediction - lag1)/lag1)*100:.2f}% vs Last Qtr"
            )
        with res_col2:
            st.info(f"The model predicts spending of **₱{prediction:,.2f}** per person given the economic conditions.")

elif model:
    st.markdown("---")
    st.subheader("ℹ️ About this Project")
    st.markdown("This app predicts discretionary spending on **Clothing & Footwear** based on economic data.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Recent Data Snapshot:**")
        st.write(f"Most Recent Year: {int(last_data['Year'])}")
        st.write(f"Last Known Spending: ₱{last_data['HFCE_Per_Capita']:,.2f}")
    with col2:
        st.write("**Model:** XGBoost Regressor")
        st.write("**Data Sources:** PSA, BSP")
