import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# Robust Import Handling
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    st.error(f"Critical Library Error: {e}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Textile Demand Predictor", layout="wide")

st.title("🤖 AI Textile Demand Prediction System")
st.markdown("""
This application uses **Random Forest** and **XGBoost** to forecast per-capita spending on clothing in the Philippines.
It processes raw economic data (Inflation, HFCE, Population, CES) to predict future demand.
""")

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename): return os.path.join(current_dir, filename)

    try:
        hfce_df = pd.read_csv(get_path('HFCE_data.csv'), header=8, encoding='latin1')
        pop_df = pd.read_csv(get_path('Population_data.csv'), encoding='latin1')
        inflation_df = pd.read_csv(get_path('Inflation_data.csv'), encoding='latin1')
        return hfce_df, pop_df, inflation_df, current_dir
    except FileNotFoundError as e:
        st.error(f"Missing File Error: {e}")
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
            # Scan up to 50 rows to find valid numeric data
            for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 50, len(df_raw))):
                try:
                    # Check column 2 (index 2) which usually holds Q1 data
                    test_val = str(df_raw.iloc[idx, 2]).replace(',', '').strip()
                    if test_val and test_val.lower() != 'nan':
                        float(test_val) # Try converting to float
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
                    if year_str.isdigit() and len(year_str) == 4: current_year = int(year_str)
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
    # 1. Clean HFCE
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year_Column', value_name='Value')
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

    hfce_target['Year'] = hfce_target['Year_Column'].map(col_to_year)
    hfce_target['Quarter'] = hfce_target['Year_Column'].map(col_to_quarter)
    
    hfce_target.drop(columns=['Year_Column'], inplace=True)
    hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear', 'Year', 'Quarter'], inplace=True)
    hfce_target['Year'] = hfce_target['Year'].astype(int)
    hfce_target['HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000 

    # 2. Clean Population
    pop_df = pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source', 'Interpolated Quarterly Estimate': 'Quarterly_Population'})
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter'] = pop_df.groupby('Year').cumcount().apply(lambda x: 'Q' + str(x % 4 + 1))
    pop_df['Year'] = pop_df['Year'].astype(int)

    # 3. Clean Inflation
    inflation_df.columns = ['Year', 'Quarter_Int', 'Inflation_Rate']
    inflation_df['Quarter'] = inflation_df['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    inflation_df['Year'] = pd.to_numeric(inflation_df['Year'], errors='coerce')
    inflation_df = inflation_df[['Year', 'Quarter', 'Inflation_Rate']].dropna()

    # 4. Extract CES
    df_ccis = extract_ces_indicator(current_dir, 'CES_Tab_1.csv', 'ROW_0_fallback', 'CCIS_Overall')
    df_fin = extract_ces_indicator(current_dir, 'CES_Tab_3.csv', 'ROW_0_fallback', 'CES_FinCondition')
    df_inc = extract_ces_indicator(current_dir, 'CES_Tab_4.csv', 'ROW_0_fallback', 'CES_Income')

    # 5. Merge
    data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
    for df in [df_ccis, df_fin, df_inc]:
        if not df.empty:
            data = pd.merge(data, df, on=['Year', 'Quarter'], how='left')
    data = pd.merge(data, inflation_df, on=['Year', 'Quarter'], how='left')

    # 6. Feature Engineering
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
    
    # Growth Rates (Safe Calculation)
    if 'Inflation_Rate' in data.columns:
        data['Inflation_Growth'] = data['Inflation_Rate'].pct_change()
    
    if 'CCIS_Overall' in data.columns:
        data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
        
    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.dropna(inplace=True)
    
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['Quarter'], drop_first=False)
    
    # Safe drop of 'Unnamed: 0' and 'Original_Column' if they exist
    cols_to_drop = ['Unnamed: 0', 'Original_Column']
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True)
    
    return data

# --- MAIN APP LOGIC ---

hfce, pop, infl, c_dir = load_data()

if hfce is not None:
    with st.spinner('Training models...'):
        df = process_pipeline(hfce, pop, infl, c_dir)
        
        target = 'HFCE_Per_Capita'
        # Define features dynamically based on what exists in DF
        exclude_cols = [target, 'HFCE_Clothing_Footwear', 'Quarterly_Population', 'Original_Column']
        features = [c for c in df.columns if c not in exclude_cols]
        
        X = df[features]
        y = df[target]
        
        # --- FINAL DATA CLEANING SAFEGUARD ---
        # Ensure absolutely all data passed to model is numeric
        for col in X.columns:
            # If column is object type (string), remove commas and convert
            if X[col].dtype == object:
                X[col] = X[col].astype(str).str.replace(',', '')
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Do the same for target y
        if y.dtype == object:
            y = y.astype(str).str.replace(',', '')
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop any rows that failed conversion (became NaN)
        # Combine temporarily to drop aligned rows
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined[target]
        
        # Chronological Split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        # Train XGBoost
        xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xg.fit(X_train, y_train)
        y_pred_xg = xg.predict(X_test)
        
        # Select Best Model
        r2_rf = r2_score(y_test, y_pred_rf)
        r2_xg = r2_score(y_test, y_pred_xg)
        
        if r2_rf > r2_xg:
            best_model = rf
            best_name = "Random Forest"
            y_pred = y_pred_rf
            best_r2 = r2_rf
        else:
            best_model = xg
            best_name = "XGBoost"
            y_pred = y_pred_xg
            best_r2 = r2_xg

    # --- DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", best_name)
    col2.metric("Accuracy (R²)", f"{best_r2:.2%}")
    col3.metric("Avg Error (MAE)", f"₱{mean_absolute_error(y_test, y_pred):.2f}")
    
    st.subheader("📊 Forecast vs. Actual (Test Set)")
    chart_data = pd.DataFrame({
        'Actual Spending': y_test.values,
        'Predicted Spending': y_pred
    })
    st.line_chart(chart_data)
    
    st.subheader("🔮 Demand Simulator")
    st.info("Enter economic data to predict clothing spending for the **next quarter**.")
    
    # Get last known data for defaults
    if not X.empty:
        last_row = X.iloc[-1]
        last_val_infl = float(last_row['Inflation_Rate']) if 'Inflation_Rate' in last_row else 5.0
        last_val_ccis = float(last_row['CCIS_Overall']) if 'CCIS_Overall' in last_row else -10.0
        last_val_lag = float(last_row['HFCE_Lag1']) if 'HFCE_Lag1' in last_row else 1500.0
    else:
        last_val_infl = 5.0
        last_val_ccis = -10.0
        last_val_lag = 1500.0

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        in_infl = c1.number_input("Inflation Rate (%)", value=last_val_infl, step=0.1)
        in_ccis = c2.number_input("Consumer Confidence Index", value=last_val_ccis, step=0.1)
        in_lag1 = c3.number_input("Prev Quarter Spending (₱)", value=last_val_lag, step=50.0)
        
        q_select = st.selectbox("Target Quarter", ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"])
        submit = st.form_submit_button("🚀 Predict Demand")
    
    if submit:
        # Construct input
        input_vector = last_row.copy()
        
        # Update available fields safely
        if 'Inflation_Rate' in input_vector: input_vector['Inflation_Rate'] = in_infl
        if 'CCIS_Overall' in input_vector: input_vector['CCIS_Overall'] = in_ccis
        if 'HFCE_Lag1' in input_vector: input_vector['HFCE_Lag1'] = in_lag1
        
        # Set Seasonality
        input_vector['Quarter_Q1'] = 1 if "Q1" in q_select else 0
        input_vector['Quarter_Q2'] = 1 if "Q2" in q_select else 0
        input_vector['Quarter_Q3'] = 1 if "Q3" in q_select else 0
        input_vector['Quarter_Q4'] = 1 if "Q4" in q_select else 0
        
        pred = best_model.predict(pd.DataFrame([input_vector]))[0]
        
        st.markdown("---")
        st.success(f"### 🎯 Forecast: **₱{pred:,.2f}** per person")
        
        # Contextual Insight
        pct_change = ((pred - in_lag1) / in_lag1) * 100
        st.caption(f"This represents a **{pct_change:+.1f}%** change from the previous quarter.")
        
    # Feature Importance Plot
    st.markdown("---")
    st.subheader(f"🔍 Feature Importance ({best_name})")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance['Feature'], importance['Importance'], color='teal')
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

else:
    st.error("Data load failed. Please check CSV files.")
