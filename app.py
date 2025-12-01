import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Robust Import Handling
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Critical Error: Matplotlib is missing. Please check requirements.txt")
    st.stop()

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    st.error(f"Scikit-learn Import Error: {e}")
    st.info("Debugging Hint: Ensure 'scikit-learn' (not 'sklearn') is in requirements.txt")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Textile Demand Forecaster", layout="centered")

st.title("🔮 Future Textile Demand Forecaster")
st.markdown("""
**Predict Per-Capita Clothing Spending** based on economic indicators.
Enter your economic outlook below to forecast demand for the next quarter.
""")

# --- DATA LOADING & PROCESSING (Hidden Logic) ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename): return os.path.join(current_dir, filename)

    try:
        hfce_df = pd.read_csv(get_path('HFCE_data.csv'), header=8, encoding='latin1')
        pop_df = pd.read_csv(get_path('Population_data.csv'), encoding='latin1')
        inflation_df = pd.read_csv(get_path('Inflation_data.csv'), encoding='latin1')
        return hfce_df, pop_df, inflation_df, current_dir
    except FileNotFoundError:
        return None, None, None, None

def extract_ces_indicator(current_dir, file_name, search_term, col_name):
    try:
        file_path = os.path.join(current_dir, file_name)
        if not os.path.exists(file_path): return pd.DataFrame()
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1')
        
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
                    val_str = str(val_year).strip().replace(',', '').replace('.0', '')
                    if val_str.isdigit() and len(val_str) == 4: current_year = int(val_str)
                except: pass
            
            val_qtr = quarter_row[col_idx]
            val_data = data_row[col_idx]
            if pd.notna(val_qtr) and pd.notna(val_data) and current_year:
                qtr_str = str(val_qtr).strip()
                if qtr_str in ['Q1', 'Q2', 'Q3', 'Q4']:
                    dates.append((current_year, qtr_str))
                    values.append(val_data)

        df_clean = pd.DataFrame(dates, columns=['Year', 'Quarter'])
        df_clean[col_name] = pd.to_numeric(pd.Series(values).astype(str).str.replace(',', ''), errors='coerce')
        return df_clean.dropna()
    except: return pd.DataFrame()

@st.cache_data
def process_pipeline(hfce_df, pop_df, inflation_df, current_dir):
    # HFCE Cleaning
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Original_Column', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    hfce_target = hfce_target.rename(columns={'Value': 'HFCE_Clothing_Footwear'})

    col_to_year = {}
    col_to_quarter = {}
    current_year = None
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']

    for i, col_name in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_name):
            try:
                val_str = str(col_name).strip()
                if val_str.isdigit() and len(val_str) == 4:
                    current_year = int(val_str)
            except: pass
        if current_year:
            col_to_year[col_name] = current_year
            col_to_quarter[col_name] = quarter_names[i % 4]

    hfce_target['Year'] = hfce_target['Original_Column'].map(col_to_year)
    hfce_target['Quarter'] = hfce_target['Original_Column'].map(col_to_quarter)
    
    hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear', 'Year', 'Quarter'], inplace=True)
    hfce_target['Year'] = hfce_target['Year'].astype(int)
    hfce_target['HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000 

    # Population Cleaning
    pop_df = pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source', 'Interpolated Quarterly Estimate': 'Quarterly_Population'})
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter'] = pop_df.groupby('Year').cumcount().apply(lambda x: 'Q' + str(x % 4 + 1))
    pop_df['Year'] = pop_df['Year'].astype(int)

    # Inflation Cleaning
    inflation_df.columns = ['Year', 'Quarter_Int', 'Inflation_Rate']
    inflation_df['Quarter'] = inflation_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inflation_df['Year'] = pd.to_numeric(inflation_df['Year'], errors='coerce')
    inflation_df = inflation_df[['Year', 'Quarter', 'Inflation_Rate']].dropna()

    # CES Extraction
    df_ccis = extract_ces_indicator(current_dir, 'CES_Tab_1.csv', 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(current_dir, 'CES_Tab_3.csv', 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(current_dir, 'CES_Tab_4.csv', 'ROW_0_fallback', 'CES_Income')

    # Merge
    data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
    for df in [df_ccis, df_fin, df_inc]:
        if not df.empty: data = pd.merge(data, df, on=['Year', 'Quarter'], how='left')
    data = pd.merge(data, inflation_df, on=['Year', 'Quarter'], how='left')

    # Feature Engineering
    data = data[data['Year'] >= 2007].sort_values(by=['Year', 'Quarter'])
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    # Lags & Rolling
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data['RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data['RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    
    data['Inflation_Growth'] = data['Inflation_Rate'].pct_change()
    data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.dropna(inplace=True)
    
    # One-Hot Encoding (Keep structure for prediction)
    data = pd.get_dummies(data, columns=['Quarter'], drop_first=False)
    if 'Unnamed: 0' in data.columns: data.drop(columns=['Unnamed: 0'], inplace=True)
    return data

# --- LOAD AND TRAIN ---
hfce, pop, infl, c_dir = load_data()

if hfce is not None:
    # Quietly train the best model in the background
    df = process_pipeline(hfce, pop, infl, c_dir)
    target = 'HFCE_Per_Capita'
    features = [c for c in df.columns if c not in [target, 'HFCE_Clothing_Footwear', 'Quarterly_Population', 'Original_Column']]
    
    X = df[features]
    y = df[target]
    
    # Using Random Forest as it proved most robust
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, 
                                  min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    
    # Get last known data for context
    last_row = X.iloc[-1]
    last_qtr_label = df.iloc[-1]['Year']
    
    # --- INPUT FORM ---
    with st.form("prediction_form"):
        st.subheader("1. Enter Economic Outlook")
        
        c1, c2 = st.columns(2)
        in_infl = c1.number_input("Expected Inflation Rate (%)", value=float(last_row['Inflation_Rate']), step=0.1)
        in_ccis = c2.number_input("Expected Consumer Confidence (Index)", value=float(last_row['CCIS_Overall']), step=0.1)
        
        st.subheader("2. Recent Spending Context")
        st.caption(f"Based on historical data, spending last quarter was ₱{last_row['HFCE_Lag1']:.2f}.")
        in_lag1 = st.number_input("Previous Quarter Spending (₱)", value=float(last_row['HFCE_Lag1']), step=50.0)
        
        # Quarter Selector for Seasonality
        st.subheader("3. Select Quarter to Predict")
        q_select = st.selectbox("Quarter", ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"])
        
        submit_btn = st.form_submit_button("🚀 Predict Demand")

    # --- PREDICTION LOGIC ---
    if submit_btn:
        # Construct input vector matching training features
        input_vector = last_row.copy()
        
        # Update with user inputs
        input_vector['Inflation_Rate'] = in_infl
        input_vector['CCIS_Overall'] = in_ccis
        input_vector['HFCE_Lag1'] = in_lag1
        
        # Approximate other lags (assuming steady state for simplicity in demo)
        input_vector['HFCE_Lag2'] = last_row['HFCE_Lag1'] 
        
        # Set seasonality flags
        input_vector['Quarter_Q1'] = 1 if "Q1" in q_select else 0
        input_vector['Quarter_Q2'] = 1 if "Q2" in q_select else 0
        input_vector['Quarter_Q3'] = 1 if "Q3" in q_select else 0
        input_vector['Quarter_Q4'] = 1 if "Q4" in q_select else 0
        
        # Predict
        prediction = model.predict(pd.DataFrame([input_vector]))[0]
        
        st.markdown("---")
        st.subheader("📊 Forecast Results")
        
        col_metrics1, col_metrics2 = st.columns(2)
        col_metrics1.metric("Predicted Spending per Person", f"₱{prediction:,.2f}")
        
        # Calculate % Change from previous
        pct_change = ((prediction - in_lag1) / in_lag1) * 100
        col_metrics2.metric("Growth vs. Previous Quarter", f"{pct_change:+.2f}%", 
                            delta_color="normal" if pct_change > 0 else "inverse")
        
        st.info(f"💡 **Insight:** A prediction of **₱{prediction:,.2f}** suggests {'strong' if prediction > 1500 else 'moderate'} demand for clothing in {q_select.split(' ')[0]}.")

else:
    st.error("Data load failed.")
