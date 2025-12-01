import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import os
import matplotlib.pyplot as plt

# --- DEPLOYMENT NOTE ---
# If deploying to Streamlit Cloud, your requirements.txt must contain:
# streamlit
# pandas
# numpy
# scikit-learn  <-- CRITICAL: Use 'scikit-learn', NOT 'sklearn'
# xgboost
# matplotlib
# pyarrow

# --- Helper Function for CES Data Extraction (Dynamic Search) ---
def extract_ces_indicator(file_path, indicator_search_term, new_column_name):
    """
    Robustly loads a CES file by searching for the 'Year'/'Quarter' structure 
    and the specific indicator row by name/keyword.
    """
    try:
        # Load the file raw to search for structure
        # header=None so we can inspect all rows
        df_raw = pd.read_csv(file_path, header=None, encoding='latin1')
        
        # 1. FIND THE QUARTER ROW (contains 'Q1', 'Q2', 'Q3', 'Q4')
        quarter_row_idx = None
        for idx, row in df_raw.iterrows():
            # Check if row contains 'Q1' AND 'Q2' (to be sure)
            row_str = row.astype(str).values
            # Join the row to string to search for Q1, Q2 easier
            row_text = " ".join(row_str)
            if 'Q1' in row_text and 'Q2' in row_text and 'Q3' in row_text:
                quarter_row_idx = idx
                break
        
        if quarter_row_idx is None:
            print(f"WARNING: Could not find Quarter labels (Q1, Q2...) in {file_path}")
            return pd.DataFrame()

        # Year row is typically directly above the Quarter row
        year_row_idx = quarter_row_idx - 1
        
        # 2. FIND THE INDICATOR DATA ROW
        # If 'ROW_0_fallback', we just take the first row AFTER the quarter row that has data
        indicator_row_idx = None
        
        if indicator_search_term == 'ROW_0_fallback':
             # Look for the first row after quarter row that has a numeric value in the 2nd column
             # We scan up to 20 rows down to find data
             for idx in range(quarter_row_idx + 1, min(quarter_row_idx + 20, len(df_raw))):
                 test_val = df_raw.iloc[idx, 2] # Check 3rd column (index 2) usually has data
                 # Try to convert to float to see if it's data
                 try:
                     # Check if it's a number (positive/negative/decimal)
                     val_str = str(test_val).replace(',', '').strip()
                     if val_str and val_str.lower() != 'nan':
                         float(val_str)
                         indicator_row_idx = idx
                         break
                 except:
                     continue
        else:
            # Search by name
            for idx, row in df_raw.iterrows():
                if idx <= quarter_row_idx: continue # Skip headers
                first_cell = str(row[0])
                if indicator_search_term.lower() in first_cell.lower():
                    indicator_row_idx = idx
                    break

        if indicator_row_idx is None:
            print(f"WARNING: Could not find valid data row for '{indicator_search_term}' in {file_path}")
            return pd.DataFrame()

        # 3. EXTRACT DATA
        # Get the actual series
        quarter_row = df_raw.iloc[quarter_row_idx]
        year_row = df_raw.iloc[year_row_idx]
        data_row = df_raw.iloc[indicator_row_idx]

        dates = []
        values = []
        current_year = None

        # Iterate through columns starting from index 1 (assuming col 0 is label)
        for col_idx in range(1, len(df_raw.columns)):
            # Update Year (Forward Fill logic manually)
            val_year = year_row[col_idx]
            if pd.notna(val_year):
                try:
                    # Clean year string (e.g. "2007")
                    year_str = str(val_year).strip().replace(',', '')
                    # Handle "2007.0" or "2007"
                    if year_str.replace('.0', '').isdigit() and len(year_str.replace('.0', '')) == 4:
                        current_year = int(year_str.replace('.0', ''))
                except:
                    pass
            
            # Get Quarter
            val_qtr = quarter_row[col_idx]
            
            # Get Data Value
            val_data = data_row[col_idx]

            if pd.notna(val_qtr) and pd.notna(val_data) and current_year is not None:
                # Basic cleanup of quarter string
                qtr_str = str(val_qtr).strip()
                if qtr_str in ['Q1', 'Q2', 'Q3', 'Q4']:
                    dates.append((current_year, qtr_str))
                    values.append(val_data)

        # Create DataFrame
        df_cleaned = pd.DataFrame(dates, columns=['Year', 'Quarter'])
        df_cleaned[new_column_name] = values
        
        # Clean Numeric Data
        df_cleaned[new_column_name] = df_cleaned[new_column_name].astype(str).str.replace(',', '')
        df_cleaned[new_column_name] = pd.to_numeric(df_cleaned[new_column_name], errors='coerce')
        
        df_cleaned.dropna(inplace=True)
        return df_cleaned

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()

# --- 1. Load Primary Data ---

# NOTE: Update this path if running locally vs Colab
GDRIVE_PATH = '/content/drive/MyDrive/' 
# If running locally in same folder, use: GDRIVE_PATH = ''

HFCE_FILE_NAME = 'HFCE_data.csv'
POP_FILE_NAME = 'Population_data.csv'
INFLATION_FILE_NAME = 'Inflation_data.csv'

hfce_file = os.path.join(GDRIVE_PATH, HFCE_FILE_NAME)
pop_file = os.path.join(GDRIVE_PATH, POP_FILE_NAME)
inflation_file = os.path.join(GDRIVE_PATH, INFLATION_FILE_NAME)

try:
    # Use latin1 to avoid codec errors
    hfce_df = pd.read_csv(hfce_file, header=8, encoding='latin1')
    pop_df = pd.read_csv(pop_file, encoding='latin1')
    df_inflation_raw = pd.read_csv(inflation_file, encoding='latin1')
except FileNotFoundError as e:
    print(f"FATAL ERROR: A required file was not found: {e}")
    raise

# --- 2. Clean HFCE Data ---
print("\n--- Cleaning HFCE Data ---")
id_col = hfce_df.columns[0]
hfce_melt = hfce_df.melt(id_vars=id_col, var_name='Year', value_name='Value')
hfce_target = hfce_melt[hfce_melt[id_col].astype(str).str.contains('Clothing and footwear', case=False, na=False)].copy()
hfce_target = hfce_target.rename(columns={'Value': 'HFCE_Clothing_Footwear'})

# Reconstruct Year/Quarter Mapping logic (Simplified for robustness)
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

# *** AGGRESSIVE NUMERIC CLEANING ***
# Force non-numeric values (like text labels) to NaN immediately
hfce_target['HFCE_Clothing_Footwear'] = pd.to_numeric(hfce_target['HFCE_Clothing_Footwear'].astype(str).str.replace(',', ''), errors='coerce')
# Drop rows that became NaN
hfce_target.dropna(subset=['HFCE_Clothing_Footwear'], inplace=True)

# *** IMPORTANT: SCALE HFCE ***
# HFCE is in "Million Pesos". Converting to "Actual Pesos" ensures reasonable numbers.
hfce_target['HFCE_Clothing_Footwear'] = hfce_target['HFCE_Clothing_Footwear'] * 1_000_000

# --- 3. Clean Population Data ---
pop_df = pop_df.rename(columns={'Annual Population Source': 'Annual_Population_Source', 
                                'Interpolated Quarterly Estimate': 'Quarterly_Population'})
pop_df['Quarterly_Population'] = pd.to_numeric(pop_df['Quarterly_Population'].astype(str).str.replace(',', ''), errors='coerce')
pop_df.dropna(subset=['Quarterly_Population'], inplace=True)
pop_df['Year'].ffill(inplace=True)
# Ensure clean Quarter column
pop_df['Quarter_Index'] = pop_df.groupby('Year').cumcount()
pop_df['Quarter'] = pop_df['Quarter_Index'].apply(lambda x: 'Q' + str(x % 4 + 1))
pop_df = pop_df[['Year', 'Quarter', 'Quarterly_Population']]
pop_df['Year'] = pop_df['Year'].astype(int)

# --- 4. Load CES Data (Using New Dynamic Search) ---
print("\n--- Extracting CES Features ---")
def get_ces_file_path(tab_num):
    return os.path.join(GDRIVE_PATH, f'CES_Tab_{tab_num}.csv')

# Use 'ROW_0_fallback' to grab the first data row found
df_ccis = extract_ces_indicator(get_ces_file_path(1), 'ROW_0_fallback', 'CCIS_Overall')
df_fin_cond = extract_ces_indicator(get_ces_file_path(3), 'ROW_0_fallback', 'CES_FinCondition')
df_income = extract_ces_indicator(get_ces_file_path(4), 'ROW_0_fallback', 'CES_Income')

# Clean Inflation
df_inflation_raw.columns = ['Year', 'Quarter_Int', 'Inflation_Annual_Static_Rate']
df_inflation_raw['Quarter'] = df_inflation_raw['Quarter_Int'].apply(lambda x: f'Q{int(x)}')
df_inflation_raw['Year'] = pd.to_numeric(df_inflation_raw['Year'], errors='coerce').astype('Int64')
df_inflation = df_inflation_raw[['Year', 'Quarter', 'Inflation_Annual_Static_Rate']].copy()
df_inflation.dropna(inplace=True)

# --- 5. Merge ---
# Merge HFCE and Population with INNER join (Mandatory)
data = pd.merge(hfce_target, pop_df, on=['Year', 'Quarter'], how='inner')
print(f"[DIAGNOSTIC] Merged HFCE & Pop. Samples: {len(data)}")

# Merge CES (Left join to check overlap)
ces_dfs = [
    (df_ccis, 'CCIS (Tab 1)'),
    (df_fin_cond, 'Financial Condition (Tab 3)'),
    (df_income, 'Income Outlook (Tab 4)')
]

for df_feature, name in ces_dfs:
    if not df_feature.empty:
        # Check overlap
        overlap = pd.merge(data, df_feature, on=['Year', 'Quarter'], how='inner')
        if len(overlap) > 0:
            print(f"MERGE SUCCESS: {name} - {len(overlap)} overlapping rows.")
            data = pd.merge(data, df_feature, on=['Year', 'Quarter'], how='left') # Keep Left to preserve dates
        else:
            print(f"MERGE WARNING: {name} has NO overlapping dates with HFCE data. Filling with NaNs.")
            data = pd.merge(data, df_feature, on=['Year', 'Quarter'], how='left')
    else:
        print(f"MERGE FAIL: {name} DataFrame is empty. Check file content.")

data = pd.merge(data, df_inflation, on=['Year', 'Quarter'], how='left')

# *** FILTER BY DATE: Only keep years where CES data likely exists (2007+) ***
print("\n--- Filtering Data to 2007-Present (CES Era) ---")
data = data[data['Year'] >= 2007]
print(f"Samples after date filtering: {len(data)}")

# Impute Missing Values (Avoid dropping everything)
data.sort_values(by=['Year', 'Quarter'], inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
data.fillna(0, inplace=True)

# Calculate Target: Spending PER PERSON
# Ensure Denominator is clean and non-zero
data = data[data['Quarterly_Population'] > 0]
data['HFCE_Per_Capita'] = data['HFCE_Clothing_Footwear'] / data['Quarterly_Population']

# --- ADVANCED FEATURE ENGINEERING (FIXED LEAKAGE) ---
# 1. Lags
data['HFCE_Lag1'] = data['HFCE_Per_Capita'].shift(1)
data['HFCE_Lag2'] = data['HFCE_Per_Capita'].shift(2)
data['HFCE_Lag4'] = data['HFCE_Per_Capita'].shift(4)

# 2. Rolling Means (FIXED: Shift BEFORE rolling to prevent seeing the answer)
# Calculate rolling mean of the PREVIOUS windows (Lag 1 + Rolling)
data['HFCE_RollingMean_2'] = data['HFCE_Per_Capita'].shift(1).rolling(window=2).mean()
data['HFCE_RollingMean_4'] = data['HFCE_Per_Capita'].shift(1).rolling(window=4).mean()

# 3. Growth Rates (Rate of Change)
data['Inflation_Growth'] = data['Inflation_Annual_Static_Rate'].pct_change()
data['CCIS_Growth'] = data['CCIS_Overall'].pct_change()

# Replace infinite values created by percentage change of 0
data.replace([np.inf, -np.inf], 0, inplace=True)

# Drop rows which will have NaNs due to shifting/rolling
data.dropna(subset=['HFCE_Lag4', 'HFCE_RollingMean_4'], inplace=True)
print(f"Samples after creating Advanced Features: {len(data)}")

# Drop intermediate columns but keep features
data.drop(columns=['HFCE_Clothing_Footwear', 'Quarterly_Population'], inplace=True)

# Clean Final Data
data = pd.get_dummies(data, columns=['Quarter'], drop_first=True)

# *** NEW CLEANING STEP ***
# Remove unnecessary index column if it exists
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

print(f"\nFinal Data Shape: {data.shape}")
print("Final Columns:", data.columns.tolist())

# --- 6. Random Forest & XGBoost ---
y = data['HFCE_Per_Capita']
X = data.drop(columns=['HFCE_Per_Capita'])

# FINAL CLEAN: Ensure all X data is numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.dropna(axis=1, how='all', inplace=True)
X.fillna(0, inplace=True) 
y = pd.to_numeric(y, errors='coerce')
valid_indices = y.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]

print(f"Training with {len(X)} clean samples.")

if len(X) == 0:
    raise ValueError("Dataset is empty after final cleaning. Check source data.")

# --- TIME SERIES SPLIT (The robust way to validate) ---
# We use TimeSeriesSplit instead of random train_test_split to avoid looking into the future
tscv = TimeSeriesSplit(n_splits=5)

# --- MODEL COMPARISON ---
print("\n--- Comparing Random Forest vs XGBoost (Time Series CV) ---")

# 1. XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='r2')
print(f"\nXGBoost Average Time-Series R²: {xgb_cv_scores.mean():.4f}")

# 2. Random Forest (Tuned)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, 
                                 min_samples_leaf=2, random_state=42)
rf_cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='r2')
print(f"Random Forest Average Time-Series R²: {rf_cv_scores.mean():.4f}")

# Select best model for final training on full data (minus test set)
# We still do a final chronological split for the evaluation metrics
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"\nFinal Chronological Split: Train={len(X_train)}, Test={len(X_test)}")

if xgb_cv_scores.mean() > rf_cv_scores.mean():
    print(">>> Selected XGBoost")
    best_model = xgb_model
    best_name = "XGBoost"
else:
    print(">>> Selected Random Forest")
    best_model = rf_model
    best_name = "Random Forest"

# Train on the chronological training set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# --- 7. Evaluation & Feature Importance (Best Model) ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Final Results ({best_name}) on Future Data ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2_score(y_test, y_pred):.4f}")

feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n--- Feature Importance ---")
print(feature_importances.sort_values(ascending=False))

# --- 8. Visualization ---
print("\n--- Generating Plots ---")

# Plot 1: Actual vs. Predicted (Time Series)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Consumption', color='blue', marker='o')
plt.plot(y_pred, label='Predicted Consumption', color='orange', linestyle='--', marker='x')
plt.title(f"Forecast vs Actual: Clothing Consumption (Future Data) - {best_name}")
plt.xlabel("Time (Quarters in Test Set)")
plt.ylabel("HFCE Per Capita (Pesos)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind='barh', color='teal')
plt.title(f"Feature Importance ({best_name})")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
