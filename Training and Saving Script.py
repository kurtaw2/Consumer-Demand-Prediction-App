import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib 
import matplotlib.pyplot as plt
# Removed: from google.colab import drive 
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration for Data Loading & Model Saving ---
# Data files are now assumed to be in the local working directory (the repository root)
DATA_FOLDER = '.' 
HFCE_FILE_NAME = 'HFCE_data.csv'
POP_FILE_NAME = 'Population_data.csv'
INFLATION_FILE_NAME = 'Inflation_data.csv'

# Output Configuration: Saved to a relative folder within the repository
MODEL_OUTPUT_FOLDER = 'HFCE_Predictor_Artifacts' 
MODEL_DIR = MODEL_OUTPUT_FOLDER # Relative path
PREDICTOR_FILENAME = 'hfce_predictor.joblib'
FEATURES_FILENAME = 'hfce_features.joblib'
DEFAULTS_FILENAME = 'hfce_defaults.joblib'
MODEL_EXPORT_PATH = os.path.join(MODEL_DIR, PREDICTOR_FILENAME)
FEATURES_EXPORT_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)
DEFAULTS_EXPORT_PATH = os.path.join(MODEL_DIR, DEFAULTS_FILENAME)

# File paths using the local DATA_FOLDER
hfce_file = os.path.join(DATA_FOLDER, HFCE_FILE_NAME)
pop_file = os.path.join(DATA_FOLDER, POP_FILE_NAME)
inflation_file = os.path.join(DATA_FOLDER, INFLATION_FILE_NAME)

def get_ces_file_path(tab_num):
    return os.path.join(DATA_FOLDER, f'CES_Tab_{tab_num}.csv')

# --- Helper Function (Data Extraction - Unchanged Logic) ---
def extract_ces_indicator(file_path, indicator_search_term, new_column_name):
    """
    Robustly loads a CES file by searching for the 'Year'/'Quarter' structure
    and the specific indicator row by name/keyword.
    """
    try:
        if not os.path.exists(file_path): 
            print(f"Warning: File not found at {file_path}")
            return pd.DataFrame()
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1') 
        
        # 1. FIND THE QUARTER ROW (contains 'Q1', 'Q2', 'Q3', 'Q4')
        quarter_row_idx = None
        for idx, row in df_raw.iterrows():
            row_str = row.astype(str).values
            row_text = " ".join(row_str)
            if 'Q1' in row_text and 'Q2' in row_text and 'Q3' in row_text:
                quarter_row_idx = idx
                break

        if quarter_row_idx is None:
            print(f"WARNING: Could not find Quarter labels (Q1, Q2...) in {file_path}")
            return pd.DataFrame()
        year_row_idx = quarter_row_idx - 1

        # 2. FIND THE INDICATOR DATA ROW
        indicator_row_idx = None
        if indicator_search_term == 'ROW_0_fallback':
             for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 20, len(df_raw))):
                 try:
                     val_str = str(df_raw.iloc[idx, 2]).replace(',', '').strip()
                     if val_str and val_str.lower() != 'nan':
                         float(val_str)
                         indicator_row_idx = idx
                         break
                 except:
                     continue
        else:
            for idx, row in df_raw.iterrows():
                if idx <= quarter_row_idx: continue
                first_cell = str(row[0])
                if indicator_search_term.lower() in first_cell.lower():
                    indicator_row_idx = idx
                    break

        if indicator_row_idx is None:
            print(f"WARNING: Could not find valid data row for '{indicator_search_term}' in {file_path}")
            return pd.DataFrame()

        # 3. EXTRACT DATA
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
                    year_str = str(val_year).strip().replace(',', '')
                    if year_str.replace('.0', '').isdigit() and len(year_str.replace('.0', '')) == 4:
                        current_year = int(year_str.replace('.0', ''))
                except:
                    pass

            val_qtr = quarter_row[col_idx]
            val_data = data_row[col_idx]

            if pd.notna(val_qtr) and pd.notna(val_data) and current_year is not None:
                qtr_str = str(val_qtr).strip()
                if qtr_str in ['Q1', 'Q2', 'Q3', 'Q4']:
                    dates.append((current_year, qtr_str))
                    values.append(val_data)

        df_cleaned = pd.DataFrame(dates, columns=['Year', 'Quarter'])
        df_cleaned[new_column_name] = values

        df_cleaned[new_column_name] = df_cleaned[new_column_name].astype(str).str.replace(',', '')
        df_cleaned[new_column_name] = pd.to_numeric(df_cleaned[new_column_name], errors='coerce')

        df_cleaned.dropna(inplace=True)
        return df_cleaned

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()

# --- Main Training and Saving Function ---

def train_and_save_model():
    # --- 1. Load Primary Data ---
    try:
        print("--- Loading primary data ---")
        hfce_df = pd.read_csv(hfce_file, header=8, encoding='latin1')
        pop_df = pd.read_csv(pop_file, encoding='latin1')
        df_inflation_raw = pd.read_csv(inflation_file, encoding='latin1')
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required file was not found: {e}")
        print(f"Please ensure all CSV files are in the folder: {DATA_FOLDER}")
        return

    # --- 2. Clean HFCE Data (same logic as user provided) ---
    print("\n--- Cleaning HFCE Data ---")
    id_col = hfce_df.columns[0]
    hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
    hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
    hfce_target = hfce_target.rename(columns={'Value': 'HFCE_Clothing_Footwear'})

    year_map = {}
    current_year = None
    for col_idx, col_value in enumerate(hfce_df.columns[1:]):
        if pd.notna(col_value):
            try:
                val_str = str(col_value).strip()
                if val_str.isdigit() and len(val_str) == 4:
                    current_year = int(val_str)
            except: pass
        year_map[col_idx + 1] = current_year

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

    hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
    hfce_target.dropna(subset=['HFCE_Clothing_Footwear'], inplace=True)
    hfce_target['HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000

    # --- 3. Clean Population Data ---
    pop_df = pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source',
                                     'Interpolated Quarterly Estimate': 'Quarterly_Population'})
    pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
    pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
    pop_df['Year'].ffill(inplace=True)
    pop_df['Quarter_Index'] = pop_df.groupby('Year').cumcount()
    pop_df['Quarter'] = pop_df['Quarter_Index'].apply(lambda x: 'Q' + str(x % 4 + 1))
    pop_df = pop_df[['Year', 'Quarter', 'Quarterly_Population']]
    pop_df['Year'] = pop_df['Year'].astype(int)

    # --- 4. Load CES Data & Inflation ---
    print("\n--- Extracting CES Features ---")
    df_ccis = extract_ces_indicator(get_ces_file_path(1), 'ROW_0_fallback', 'CCIS_Overall')
    df_fin_cond = extract_ces_indicator(get_ces_file_path(3), 'ROW_0_fallback', 'CES_FinCondition')
    df_income = extract_ces_indicator(get_ces_file_path(4), 'ROW_0_fallback', 'CES_Income')

    df_inflation_raw.columns = ['Year', 'Quarter_Int', 'Inflation_Annual_Static_Rate']
    df_inflation_raw['Quarter'] = df_inflation_raw['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
    df_inflation_raw['Year'] = pd.to_numeric(df_inflation_raw['Year'], errors='coerce').astype('Int64')
    df_inflation = df_inflation_raw[['Year', 'Quarter', 'Inflation_Annual_Static_Rate']].copy()
    df_inflation.dropna(inplace=True)

    # --- 5. Merge, Filter, and Feature Engineer ---
    data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
    
    ces_dfs = [
        (df_ccis, 'CCIS (Tab 1)'),
        (df_fin_cond, 'Financial Condition (Tab 3)'),
        (df_income, 'Income Outlook (Tab 4)')
    ]
    for df_feature, name in ces_dfs:
        if not df_feature.empty:
            data = pd.merge(data, df_feature, on=['Year', 'Quarter'], how='left')
        else:
            print(f"MERGE FAIL: {name} DataFrame is empty.")
    data = pd.merge(data, df_inflation, on=['Year', 'Quarter'], how='left')

    data = data[data['Year'] >= 2007]

    data.sort_values(by=['Year', 'Quarter'], inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)

    data = data[data['Quarterly_Population'] > 0]
    data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']

    # --- CAPTURE LAST KNOWN DATA BEFORE FEATURE ENGINEERING (for app defaults) ---
    # We capture the last row *before* feature engineering creates NaNs
    last_known_row_pre_fe = data.iloc[-1].copy()
    
    # Feature Engineering
    data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
    data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
    data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)
    data['HFCE_RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
    data['HFCE_RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()
    data['Inflation_Growth'] = data['Inflation_Annual_Static_Rate'].pct_change()
    data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()
    data.replace([np.inf, -np.inf], 0, inplace=True)

    # Drop NaNs created by shifting/rolling
    data.dropna(subset=['HFCE_Lag4', 'HFCE_RollingMean_4'], inplace=True)
    data.drop(columns=['HFCE_Clothing_Footwear', 'Quarterly_Population'], inplace=True)
    
    # --- Final Pre-processing of Last Row for Accurate Defaults ---
    # Recalculate lags and growth rates for the last *valid* row of data for defaults
    # Find the row in the fully processed 'data' just before the last row (this is the true source of prediction defaults)
    last_known_row_post_fe = data.iloc[-1]
    
    # --- 6. Final Data Preparation ---
    data = pd.get_dummies(data, columns=['Quarter'], drop_first=True)
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    y = data['HFCE_Per_Capita']
    X = data.drop(columns=['HFCE_Per_Capita'])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.dropna(axis=1, how='all', inplace=True)
    X.fillna(0, inplace=True)
    y = pd.to_numeric(y, errors='coerce')
    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    if len(X) == 0:
        print("Final dataset is empty. Cannot train model.")
        return

    # --- 7. Model Training and Selection ---
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                                     min_samples_leaf=2, random_state=42)
    
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='r2')
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='r2')

    if xgb_cv_scores.mean() > rf_cv_scores.mean():
        print(f"\n>>> Selected XGBoost (R²: {xgb_cv_scores.mean():.4f})")
        best_model = xgb_model
    else:
        print(f"\n>>> Selected Random Forest (R²: {rf_cv_scores.mean():.4f})")
        best_model = rf_model

    # Train the FINAL model on the ENTIRE clean dataset (X, y)
    print("Training final model on entire dataset for production export...")
    best_model.fit(X, y)
    
    # --- 8. Model Saving and Defaults Export ---
    try:
        # Define the dictionary of dynamic defaults for the UI
        dynamic_defaults = {
            # The most recent known historical data point:
            'Year': int(last_known_row_post_fe['Year']),
            'HFCE_Per_Capita': float(y.iloc[-1]), # The actual last predicted target value from the training data
            'HFCE_Lag1': float(last_known_row_post_fe['HFCE_Per_Capita']), # Last quarter's final consumption
            'HFCE_Lag2': float(last_known_row_post_fe['HFCE_Lag1']), 
            'HFCE_Lag4': float(last_known_row_post_fe['HFCE_Lag4']),
            'CCIS_Overall': float(last_known_row_post_fe['CCIS_Overall']),
            'CES_FinCondition': float(last_known_row_post_fe['CES_FinCondition']),
            'CES_Income': float(last_known_row_post_fe['CES_Income']),
            'Inflation_Annual_Static_Rate': float(last_known_row_post_fe['Inflation_Annual_Static_Rate']),
        }
        
        # Create the models directory in the local repo if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the trained model
        joblib.dump(best_model, MODEL_EXPORT_PATH)
        
        # Save the feature column list (CRITICAL for prediction)
        joblib.dump(X.columns.tolist(), FEATURES_EXPORT_PATH)

        # Save the dynamic defaults
        joblib.dump(dynamic_defaults, DEFAULTS_EXPORT_PATH)
        
        print("\n" + "-" * 60)
        print(f"✅ MODEL EXPORT SUCCESSFUL (Local Repo Paths)")
        print(f"Trained Model saved to: {MODEL_EXPORT_PATH}")
        print(f"Feature List saved to: {FEATURES_EXPORT_PATH}")
        print(f"Dynamic Defaults saved to: {DEFAULTS_EXPORT_PATH}")
        print("-" * 60)
        
    except Exception as e:
        print(f"\nFATAL ERROR during model serialization/export: {e}")

if __name__ == '__main__':
    # You should run this script once in your local repository environment.
    # The output folder 'HFCE_Predictor_Artifacts' will be created locally.
    train_and_save_model()
