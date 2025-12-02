import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="HFCE Prediction Tool", layout="wide")

st.title("🔮 HFCE Clothing Consumption Predictor")
st.markdown("""
**How to use:** Enter the economic indicators and historical context below to predict 
the Household Final Consumption Expenditure (Per Capita) for Clothing & Footwear.
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
    """
    Loads data, trains the best model, and returns the model + feature columns.
    """
    # 1. Load & Clean
    try:
        hfce_df = pd.read_csv(FILES['hfce'], header=8, encoding='latin1')
        pop_df = pd.read_csv(FILES['pop'], encoding='latin1')
        inf_df = pd.read_csv(FILES['inf'], encoding='latin1')
    except:
        return None, None

    # Clean HFCE
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    
    # Year/Quarter Mapping
    year_map = {}
    current_year = None
    for col_idx, col_value in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_value):
            s = str(col_value).strip()
            if s.isdigit() and len(s)==4: current_year = int(s)
        year_map[col_idx + 1] = current_year
    
    hfce_target['Real_Year'] = [year_map.get(hfce_df.columns.get_loc(row['Year'])) for i, row in hfce_target.iterrows()]
    hfce_target['Quarter'] = [['Q1','Q2','Q3','Q4'][(hfce_df.columns.get_loc(row['Year']) - 1) % 4] for i, row in hfce_target.iterrows()]
    hfce_target['Year'] = hfce_target['Real_Year']
    hfce_target.dropna(subset=['Year', 'Quarter'], inplace=True)
    hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['Value'].astype(str).str.replace(',', ''), errors='coerce') * 1_000_000
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear'], inplace=True)

    # Clean Pop
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Interpolated Quarterly Estimate'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter'] = pop_df.groupby('Year').cumcount().apply(lambda x: f'Q{x%4+1}')
    
    # Clean Inf
    inf_df.columns = ['Year', 'Quarter_Int', 'Inflation_Annual_Static_Rate']
    inf_df['Quarter'] = inf_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inf_df['Year'] = pd.to_numeric(inf_df['Year'], errors='coerce')
    
    # CES
    df_ccis = extract_ces_indicator(FILES['ces1'], 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(FILES['ces3'], 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(FILES['ces4'], 'ROW_0_fallback', 'CES_Income')

    # Merge
    data = pd.merge(hfce_target[['Year','Quarter','HFCE_Clothing_Footwear']], pop_df[['Year','Quarter','Quarterly_Population']], on=['Year','Quarter'], how='inner')
    for d in [df_ccis, df_fin, df_inc]:
        if not d.empty: data = pd.merge(data, d, on=['Year','Quarter'], how='left')
    data = pd.merge(data, inf_df[['Year','Quarter','Inflation_Annual_Static_Rate']], on=['Year','Quarter'], how='left')

    # Engineering
    data = data.sort_values(by=['Year', 'Quarter']).fillna(method='ffill').fillna(0)
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    # Lags/Rolling
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data['HFCE_RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data['HFCE_RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    data['Inflation_Growth'] = data['Inflation_Annual_Static_Rate'].pct_change()
    data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
    
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.dropna(inplace=True)
    
    # Final ML Setup
    y = data['HFCE_Per_Capita']
    X = data.drop(columns=['HFCE_Per_Capita', 'HFCE_Clothing_Footwear', 'Quarterly_Population'])
    if 'Unnamed: 0' in X.columns: X.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Get Dummies manually to ensure column consistency
    X = pd.get_dummies(X, columns=['Quarter'], drop_first=True)
    
    # Ensure numeric
    for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Train XGBoost (usually performs best)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# --- Main Application Logic ---

# 1. Train/Load Model (Hidden)
with st.spinner("Initializing Model..."):
    model, feature_cols = train_model()

if model is None:
    st.error("Could not load data. Please check repository files.")
    st.stop()

# 2. Input Form
st.sidebar.header("1. Scenario Input")
with st.sidebar.form("prediction_form"):
    st.subheader("Time")
    input_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=2025)
    input_qtr = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"])
    
    st.subheader("Economic Indicators")
    st.info("Enter your forecasted values for the target quarter.")
    val_ccis = st.number_input("Consumer Confidence Index (CCIS)", value=-10.0)
    val_fin = st.number_input("Financial Condition Index", value=-5.0)
    val_inc = st.number_input("Income Outlook Index", value=5.0)
    
    st.subheader("Inflation Context")
    val_inf_curr = st.number_input("Inflation Rate (Target Qtr) %", value=4.5)
    val_inf_prev = st.number_input("Inflation Rate (Previous Qtr) %", value=4.2, help="Used to calculate growth rate")

    st.subheader("Historical Consumption")
    st.warning("These are crucial for time-series predictions.")
    lag1 = st.number_input("Consumption Last Qtr (Lag 1)", value=1500.0)
    lag2 = st.number_input("Consumption 2 Qtrs Ago (Lag 2)", value=1450.0)
    lag4 = st.number_input("Consumption 1 Year Ago (Lag 4)", value=1400.0)

    # Submit
    submit_btn = st.form_submit_button("Calculated Prediction")

# 3. Prediction Logic
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
    # Rolling Mean 2 = (Lag1 + Lag2) / 2 approx (Technically Lag1 is the prev, Lag 2 is prev-prev)
    input_data['HFCE_RollingMean_2'] = (lag1 + lag2) / 2
    
    # Rolling Mean 4 approx
    input_data['HFCE_RollingMean_4'] = (lag1 + lag2 + lag4 + lag4) / 4 # Rough approx using available inputs
    
    # Growth Rates
    pct_change_inf = (val_inf_curr - val_inf_prev) / val_inf_prev if val_inf_prev != 0 else 0
    input_data['Inflation_Growth'] = pct_change_inf
    
    # We can't easily calc CCIS growth without prev CCIS, so we assume 0 or ask for it. 
    # For simplicity, we set 0.05 (slight growth) or 0
    input_data['CCIS_Growth'] = 0.0 

    # One-Hot Encoding for Quarter
    # Columns expected: Quarter_Q2, Quarter_Q3, Quarter_Q4 (Q1 is reference/dropped)
    input_data['Quarter_Q2'] = 1 if input_qtr == 'Q2' else 0
    input_data['Quarter_Q3'] = 1 if input_qtr == 'Q3' else 0
    input_data['Quarter_Q4'] = 1 if input_qtr == 'Q4' else 0
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Align columns perfectly with training data (add missing cols as 0, remove extra)
    df_final = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        if col in df_input.columns:
            df_final.loc[0, col] = df_input.loc[0, col]
        else:
            df_final.loc[0, col] = 0
            
    # PREDICT
    prediction = model.predict(df_final)[0]
    
    # 4. Display Output
    st.divider()
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(label="Predicted HFCE Per Capita", value=f"₱ {prediction:,.2f}")
    
    with col_res2:
        st.success("Prediction Generated!")
        st.write("Based on the indicators provided, the model predicts the average spending per person on Clothing & Footwear.")
        
        # Simple Interpretation
        growth = ((prediction - lag1) / lag1) * 100
        if growth > 0:
            st.write(f"📈 This represents a **{growth:.2f}% increase** from the previous quarter.")
        else:
            st.write(f"📉 This represents a **{abs(growth):.2f}% decrease** from the previous quarter.")

else:
    st.info("👈 Please enter the scenario details in the sidebar and click 'Calculated Prediction'")
    
    # Optional: Show Feature Importance just so the page isn't empty
    if model:
        st.subheader("Model Insights")
        st.write("While you wait, here are the factors driving the model's decisions:")
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(5)
        st.bar_chart(feat_imp)
