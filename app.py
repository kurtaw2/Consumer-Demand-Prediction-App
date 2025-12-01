import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Textile Demand Predictor", layout="wide")

st.title("🤖 AI Textile Demand Prediction System")
st.markdown("""
This application uses a **Random Forest** model to predict per-capita spending on clothing and footwear in the Philippines.
It integrates economic indicators (Inflation) and consumer sentiment (CES) to forecast demand.
""")

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_and_process_data():
    # 1. Load Data (Robust Path Handling for Cloud)
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Helper to get full path
        def get_path(filename):
            return os.path.join(current_dir, filename)

        # Check if files exist
        required_files = ['HFCE_data.csv', 'Population_data.csv', 'Inflation_data.csv', 
                          'CES_Tab_1.csv', 'CES_Tab_3.csv', 'CES_Tab_4.csv']
        
        for f in required_files:
            if not os.path.exists(get_path(f)):
                st.error(f"❌ Critical file not found: {f}")
                st.info("Ensure all CSV files are uploaded to the same GitHub folder as this script.")
                return None
        
        hfce_df = pd.read_csv(get_path('HFCE_data.csv'), header=8, encoding='latin1')
        pop_df = pd.read_csv(get_path('Population_data.csv'), encoding='latin1')
        inflation_df = pd.read_csv(get_path('Inflation_data.csv'), encoding='latin1')
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None

    # --- 2. CLEANING HFCE ---
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    hfce_target = hfce_target.rename(columns={'Value': 'HFCE_Clothing_Footwear'})

    # Year/Quarter Mapping
    year_map = {}
    for col_idx, col_value in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_value):
            try:
                val_str = str(col_value).strip()
                if val_str.isdigit() and len(val_str) == 4:
                    current_year = int(val_str)
            except: pass
        year_map[col_idx + 1] = current_year if 'current_year' in locals() else None

    year_labels = []
    quarter_labels = []
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for i, row in hfce_target.iterrows():
        col_index = hfce_df.columns.get_loc(row['Year'])
        year = year_map.get(col_index)
        quarter = quarter_names[(col_index - 1) % 4]
        year_labels.append(year)
        quarter_labels.append(quarter)

    hfce_target['Year'] = year_labels
    hfce_target['Quarter'] = quarter_labels
    hfce_target.dropna(subset=['Year', 'Quarter'], inplace=True)
    hfce_target['Year'] = hfce_target['Year'].astype(int)
    
    # Numeric Conversion & Scaling
    hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear'], inplace=True)
    hfce_target['HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000 # Scale to Pesos

    # --- 3. CLEANING POPULATION ---
    pop_df = pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source', 
                                    'Interpolated Quarterly Estimate': 'Quarterly_Population'})
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter_Index'] = pop_df.groupby('Year').cumcount()
    pop_df['Quarter'] = pop_df['Quarter_Index'].apply(lambda x: 'Q' + str(x % 4 + 1))
    pop_df = pop_df[['Year', 'Quarter', 'Quarterly_Population']]
    pop_df['Year'] = pop_df['Year'].astype(int)

    # --- 4. CES EXTRACTION HELPER ---
    def extract_ces(file_name, search_term, col_name):
        try:
            # Load with robust path
            full_path = os.path.join(current_dir, file_name)
            df_raw = pd.read_csv(full_path, header=None, encoding='latin1')
            quarter_row_idx = None
            for idx, row in df_raw.iterrows():
                row_text = " ".join(row.astype(str).values)
                if 'Q1' in row_text and 'Q2' in row_text:
                    quarter_row_idx = idx
                    break
            
            if quarter_row_idx is None: return pd.DataFrame()

            year_row_idx = quarter_row_idx - 1
            indicator_row_idx = None
            
            if search_term == 'ROW_0_fallback':
                 for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 20, len(df_raw))):
                     try:
                         float(str(df_raw.iloc[idx, 2]).replace(',', ''))
                         indicator_row_idx = idx
                         break
                     except: continue
            
            if indicator_row_idx is None: return pd.DataFrame()

            quarter_row = df_raw.iloc[quarter_row_idx]
            year_row = df_raw.iloc[year_row_idx]
            data_row = df_raw.iloc[indicator_row_idx]

            dates, values = [], []
            current_year = None
            
            for col_idx in range(1, len(df_raw.columns)):
                val_year = year_row[col_idx]
                if pd.notna(val_year):
                    try:
                        year_str = str(val_year).strip().replace(',', '').replace('.0', '')
                        if year_str.isdigit() and len(year_str) == 4:
                            current_year = int(year_str)
                    except: pass
                
                val_qtr = quarter_row[col_idx]
                val_data = data_row[col_idx]

                if pd.notna(val_qtr) and pd.notna(val_data) and current_year is not None:
                    qtr_str = str(val_qtr).strip()
                    if qtr_str in ['Q1', 'Q2', 'Q3', 'Q4']:
                        dates.append((current_year, qtr_str))
                        values.append(val_data)

            df_clean = pd.DataFrame(dates, columns=['Year', 'Quarter'])
            df_clean[col_name] = pd.to_numeric(pd.Series(values).astype(str).str.replace(',', ''), errors='coerce')
            return df_clean.dropna()
        except: return pd.DataFrame()

    # Load CES Tabs
    df_ccis = extract_ces('CES_Tab_1.csv', 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces('CES_Tab_3.csv', 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces('CES_Tab_4.csv', 'ROW_0_fallback', 'CES_Income')

    # Clean Inflation
    inflation_df.columns = ['Year', 'Quarter_Int', 'Inflation_Rate']
    inflation_df['Quarter'] = inflation_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inflation_df['Year'] = pd.to_numeric(inflation_df['Year'], errors='coerce')
    inflation_df = inflation_df[['Year', 'Quarter', 'Inflation_Rate']].dropna()

    # --- 5. MERGE ALL ---
    data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
    
    # Merge Features
    for df in [df_ccis, df_fin, df_inc]:
        if not df.empty:
            data = pd.merge(data, df, on=['Year', 'Quarter'], how='left')
    
    data = pd.merge(data, inflation_df, on=['Year', 'Quarter'], how='left')

    # Filter & Impute
    data = data[data['Year'] >= 2007] # CES Start
    data = data.sort_values(by=['Year', 'Quarter'])
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    # Calculate Target
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    # --- FEATURE ENGINEERING ---
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    # FIX: Shift before rolling to prevent leakage
    data['RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    
    data.dropna(inplace=True)
    
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['Quarter'], drop_first=False) # Keep all quarters for display
    
    return data

# --- MAIN APP LOGIC ---

df = load_and_process_data()

if df is not None:
    # Sidebar Controls
    st.sidebar.header("🛠️ Model Controls")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30) / 100
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200)
    
    # Split Data
    # Identify feature columns (exclude non-numeric or target)
    target = 'HFCE_Per_Capita'
    features = [c for c in df.columns if c not in [target, 'HFCE_Clothing_Footwear', 'Quarterly_Population']]
    
    # Separate Train/Test chronologically
    split_idx = int(len(df) * (1 - test_size))
    X = df[features]
    y = df[target]
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train Model (Using best params found earlier)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, 
                               min_samples_split=5, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # --- METRICS DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score (Accuracy)", f"{r2_score(y_test, y_pred):.2f}")
    col2.metric("MAE (Avg Error)", f"₱{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("RMSE", f"₱{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    
    # --- VISUALIZATIONS ---
    st.subheader("📈 Consumption Forecast vs. Actual")
    
    # Create comparison dataframe for chart
    chart_data = pd.DataFrame({
        'Actual Spending': y_test.values,
        'Predicted Spending': y_pred
    }, index=df.iloc[split_idx:]['Year'].astype(str)) # Use Year as index for chart
    
    st.line_chart(chart_data)
    
    # Feature Importance
    st.subheader("🔍 What Drives Demand?")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance.set_index('Feature'))
    
    # --- SIMULATOR ---
    st.markdown("---")
    st.subheader("🔮 Demand Simulator")
    st.write("Adjust economic factors to see how they impact predicted clothing spending.")
    
    c1, c2, c3 = st.columns(3)
    user_inflation = c1.number_input("Inflation Rate (%)", value=5.0)
    user_sentiment = c2.number_input("Consumer Confidence Index", value=-10.0)
    user_lag = c3.number_input("Last Quarter's Spending (₱)", value=1500.0)
    
    # Create a dummy row for prediction (Simplified)
    # Ideally, we would need to reconstruct all lags, but for a simple demo
    # we take the last known state and update the user inputs.
    input_data = X_test.iloc[-1:].copy() 
    
    # Map user inputs to features if they exist
    if 'Inflation_Rate' in input_data.index: input_data['Inflation_Rate'] = user_inflation
    if 'CCIS_Overall' in input_data.index: input_data['CCIS_Overall'] = user_sentiment
    if 'HFCE_Lag1' in input_data.index: input_data['HFCE_Lag1'] = user_lag
    
    sim_pred = rf.predict([input_data])[0]
    st.success(f"Estimated Spending: **₱{sim_pred:,.2f}** per person")

else:
    st.warning("Data could not be loaded. Please check your CSV files.")
