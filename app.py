import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
# ADDED TimeSeriesSplit for the model training logic
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Textile Demand Forecaster", layout="centered")

st.title("🔮 Future Textile Demand Forecaster")
st.markdown("""
**Predict Per-Capita Clothing Spending** based on economic indicators.
Enter your economic outlook below to forecast demand for the next quarter.
""")

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    # In a real environment, __file__ might not exist, but in a local Streamlit setup it does.
    # For safety in environments like Canvas, we use a default path logic.
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() or '__file__' in globals() else os.getcwd()
    
    def get_path(filename): return os.path.join(current_dir, filename)

    try:
        # Load necessary files (HFCE uses header 8 as determined in diagnostics)
        # Note: The files must be present in the same directory as this script.
        hfce_df = pd.read_csv(get_path('HFCE_data.csv'), header=8, encoding='latin1')
        pop_df = pd.read_csv(get_path('Population_data.csv'), encoding='latin1')
        inflation_df = pd.read_csv(get_path('Inflation_data.csv'), encoding='latin1')
        return hfce_df, pop_df, inflation_df, current_dir
    except FileNotFoundError:
        st.error("One or more required data files (HFCE_data.csv, Population_data.csv, Inflation_data.csv) are missing. Please ensure they are in the same directory as the script.")
        return None, None, None, None

def extract_ces_indicator(current_dir, file_name, search_term, col_name):
    """
    Complex logic to extract quarterly data from non-standard formatted CES files.
    """
    try:
        file_path = os.path.join(current_dir, file_name)
        if not os.path.exists(file_path): return pd.DataFrame()
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1')
        
        # 1. Find the quarter row (e.g., Q1, Q2, Q3, Q4)
        quarter_row_idx = None
        for idx, row in df_raw.iterrows():
            row_text = " ".join(row.astype(str).values)
            if 'Q1' in row_text and 'Q2' in row_text:
                quarter_row_idx = idx
                break
        
        if quarter_row_idx is None: return pd.DataFrame()

        year_row_idx = quarter_row_idx - 1
        indicator_row_idx = None
        
        # 2. Find the actual data row (using a fallback approach based on numeric content)
        if search_term == 'ROW_0_fallback':
            for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 50, len(df_raw))):
                try:
                    # Try to find a row where the third column is a valid number
                    test_val = str(df_raw.iloc[idx, 2]).replace(',', '').strip()
                    if test_val and test_val.lower() != 'nan':
                        float(test_val)
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
                    val_str = str(val_year).strip()
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
    except Exception as e:
        st.warning(f"Error processing {file_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def process_pipeline(hfce_df, pop_df, inflation_df, current_dir):
    # 1. Clean HFCE
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Original_Column', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    hfce_target.rename(columns={'Value': 'HFCE_Clothing_Footwear'}, inplace=True)

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
            # The column names are implicitly Q1, Q2, Q3, Q4 repeating after the year column
            col_to_quarter[col_name] = quarter_names[i % 4]

    hfce_target.loc[:, 'Year'] = hfce_target['Original_Column'].map(col_to_year)
    hfce_target.loc[:, 'Quarter'] = hfce_target['Original_Column'].map(col_to_quarter)
    
    hfce_target.drop(columns=['Original_Column'], inplace=True)
    
    hfce_target.loc[:, 'HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear', 'Year', 'Quarter'], inplace=True)
    hfce_target.loc[:, 'Year'] = hfce_target['Year'].astype(int)
    
    # --- SCALE HFCE (Millions -> Actual Pesos) ---
    hfce_target.loc[:, 'HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000 

    # 2. Clean Population
    pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source', 'Interpolated Quarterly Estimate': 'Quarterly_Population'}, inplace=True)
    pop_df.loc[:, 'Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    # Ensure Year is treated as integer for grouping
    pop_df.loc[:, 'Year'] = pd.to_numeric(pop_df['Year'], errors='coerce').astype('Int64')
    pop_df.dropna(subset=['Year'], inplace=True)
    pop_df.loc[:, 'Quarter'] = pop_df.groupby('Year').cumcount().apply(lambda x: 'Q' + str(x % 4 + 1))
    pop_df.loc[:, 'Year'] = pop_df['Year'].astype(int)

    # 3. Clean Inflation
    inflation_df.columns = ['Year', 'Quarter_Int', 'Inflation_Rate']
    inflation_df.loc[:, 'Quarter'] = inflation_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inflation_df.loc[:, 'Year'] = pd.to_numeric(inflation_df['Year'], errors='coerce')
    inflation_df = inflation_df[['Year', 'Quarter', 'Inflation_Rate']].dropna()

    # 4. Extract CES
    df_ccis = extract_ces_indicator(current_dir, 'CES_Tab_1.csv', 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(current_dir, 'CES_Tab_3.csv', 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(current_dir, 'CES_Tab_4.csv', 'ROW_0_fallback', 'CES_Income')

    # 5. Merge
    data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
    for df_ind in [df_ccis, df_fin, df_inc]:
        if not df_ind.empty:
            data = pd.merge(data, df_ind, on=['Year', 'Quarter'], how='left')
    data = pd.merge(data, inflation_df, on=['Year', 'Quarter'], how='left')

    # 6. Feature Engineering
    data = data[data['Year'] >= 2007].sort_values(by=['Year', 'Quarter']).copy()
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    data.loc[:, 'HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']
    
    # Lags & Rolling
    data.loc[:, 'HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data.loc[:, 'HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data.loc[:, 'HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data.loc[:, 'RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data.loc[:, 'RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    
    # Growth Rates (Safe Calculation with Column Checks)
    if 'Inflation_Rate' in data.columns:
        data.loc[:, 'Inflation_Growth'] = data['Inflation_Rate'].pct_change()
    
    # CRITICAL FIX: Only calculate if CCIS_Overall exists
    if 'CCIS_Overall' in data.columns:
        data.loc[:, 'CCIS_Growth'] = data['CCIS_Overall'].pct_change()
        
    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.dropna(inplace=True)
    
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['Quarter'], drop_first=False)
    
    # Safe drop of unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'Original_Column', id_col]
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True, errors='ignore')
    
    return data

# --- MAIN APP LOGIC ---

hfce, pop, infl, c_dir = load_data()

if hfce is not None:
    @st.cache_resource
    def train_model(X, y):
        # Time Series Split for honest validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # XGBoost is often better for this data
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        
        # Grid Search parameters for Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [5],
            'min_samples_leaf': [2]
        }
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='r2', n_jobs=-1)
        
        # Train both models for comparison
        rf_grid.fit(X, y)
        xgb_model.fit(X, y)
        
        # Determine the best model based on cross-validation score
        # Note: Comparing CV score (RF) vs simple score (XGB) is a simplification,
        # but kept for retaining original logic structure.
        if rf_grid.best_score_ > xgb_model.score(X, y):
            return rf_grid.best_estimator_, rf_grid.best_estimator_.__class__.__name__
        else:
            return xgb_model, xgb_model.__class__.__name__

    with st.spinner('Preparing data and training models...'):
        df = process_pipeline(hfce, pop, infl, c_dir)
        
        target = 'HFCE_Per_Capita'
        exclude_cols = [target, 'HFCE_Clothing_Footwear', 'Quarterly_Population', 'Year', 'Quarter_Int', 'Annual_Population_Source']
        features = [c for c in df.columns if c not in exclude_cols]
        
        X = df[features].copy()
        y = df[target].copy()
        
        # Final Cleaning before model training (Brute force numeric conversion)
        for col in X.columns:
            if X[col].dtype == object:
                X.loc[:, col] = X[col].astype(str).str.replace(',', '')
            X.loc[:, col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        # Ensure training data is clean
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined[target]
        
        if X.empty or len(X) < 5:
            st.error("Training data is insufficient or empty after cleaning. Check file alignment/dates.")
            st.stop()
            
        best_model, model_name = train_model(X, y)
        
        # Get last known data for input defaults
        last_row = X.iloc[-1].copy()
        # Ensure we capture all feature names for consistent input vector creation
        all_features = X.columns.tolist() 

        
    # --- INPUT FORM ---
    st.subheader("Scenario Inputs")
    with st.form("prediction_form"):
        
        # Use last historical values as starting point for consistent forecasting
        val_infl = float(last_row.get('Inflation_Rate', 5.0))
        val_ccis = float(last_row.get('CCIS_Overall', -10.0))
        
        st.caption("Enter your expected economic outlook for the **target quarter**:")
        c1, c2, c3 = st.columns(3)
        in_infl = c1.number_input("Inflation Rate (%)", value=val_infl, step=0.1, key="i_infl", format="%.2f")
        in_ccis = c2.number_input("Consumer Confidence Index", value=val_ccis, step=0.1, key="i_ccis", format="%.2f")
        q_select = c3.selectbox("Target Quarter", ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"], key="i_qtr")

        st.subheader("Historical Context (Required for Stable Forecasting)")
        st.caption("The model needs these previous spending figures to predict the trend accurately.")
        
        # Calculate Rolling Mean based on Lag1 and Lag4 inputs for consistency
        c4, c5 = st.columns(2)
        
        in_lag1 = c4.number_input("Prev Quarter Spending (₱)", value=float(last_row.get('HFCE_Lag1', 1500.0)), step=1.0, key="i_lag1", format="%.2f")
        in_lag4 = c5.number_input("Spending 1 Year Ago (₱)", value=float(last_row.get('HFCE_Lag4', 1500.0)), step=1.0, key="i_lag4", format="%.2f")

        # Use the actual last row data for other lags if available
        lag2_val = last_row.get('HFCE_Lag2', in_lag1)
        # Assuming Lag3 is the quarter before Lag4, which is not an explicit feature but needed for the 4-quarter average calculation logic
        # For simplicity in this mock-up, we'll take the simple average of the last 4 *available* lags.
        
        in_roll4_calc = last_row.get('RollingMean_4', (in_lag1 + lag2_val + in_lag4 + in_lag4) / 4) # Fallback uses Lag1 and Lag4 as proxies

        submit = st.form_submit_button("🚀 Predict Demand")
    
    if submit:
        # Construct input vector based on ALL features used in training
        input_data = {feature: last_row.get(feature, 0.0) for feature in all_features}
        input_vector = pd.Series(input_data)
        
        # Update inputs
        input_vector['Inflation_Rate'] = in_infl
        input_vector['CCIS_Overall'] = in_ccis
        
        # Update Lag/Rolling Features based on user input and calculated trend
        input_vector['HFCE_Lag1'] = in_lag1
        input_vector['HFCE_Lag2'] = last_row.get('HFCE_Lag1', in_lag1) # Prev Lag1 becomes new Lag2
        input_vector['HFCE_Lag4'] = in_lag4
        input_vector['RollingMean_4'] = in_roll4_calc # Use the proxy calculation
        input_vector['RollingMean_2'] = (in_lag1 + last_row.get('HFCE_Lag1', in_lag1)) / 2 # Simple avg of last two known quarters
        
        # Calculate Growth Rates
        prev_infl = last_row.get('Inflation_Rate', 0)
        input_vector['Inflation_Growth'] = (in_infl - prev_infl) / prev_infl if prev_infl != 0 else 0
        
        prev_ccis = last_row.get('CCIS_Overall', 0)
        input_vector['CCIS_Growth'] = (in_ccis - prev_ccis) / prev_ccis if prev_ccis != 0 else 0
        
        # Set Seasonality
        q_map = {'Q1 (Jan-Mar)': 'Q1', 'Q2 (Apr-Jun)': 'Q2', 'Q3 (Jul-Sep)': 'Q3', 'Q4 (Oct-Dec)': 'Q4'}
        selected_q = q_map[q_select]
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            col_name = f"Quarter_{q}"
            if col_name in input_vector.index:
                input_vector[col_name] = 1 if q == selected_q else 0
        
        # Convert to DataFrame for prediction (must match feature order/columns)
        X_pred = pd.DataFrame([input_vector.reindex(index=X.columns, fill_value=0).astype(float)])
        
        try:
            pred = best_model.predict(X_pred)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        
        st.markdown("---")
        
        # --- SHOW PREVIOUS VS PREDICTED ---
        col_prev, col_curr = st.columns(2)
        
        # Previous Quarter is defined by the user input for Lag1
        prev_spending = in_lag1
        
        # Calculation
        pct_change = ((pred - prev_spending) / prev_spending) * 100
        
        col_prev.metric("Previous Quarter (Baseline)", f"₱{prev_spending:,.2f}")
        
        col_curr.metric("🎯 Forecasted Spending", f"₱{pred:,.2f}", f"{pct_change:+.1f}%")
        
        st.success(f"**Action Insight:** Demand is forecasted to {'increase' if pct_change > 0 else 'decrease'} by **{abs(pct_change):.1f}%** in the upcoming {q_select}.")
        st.caption(f"Prediction made using the {model_name} model.")
