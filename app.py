# ... (Keep all imports and helper functions the same as before) ...

# --- Main Application Logic ---

# 1. Train/Load Model (Hidden in background)
with st.spinner("Initializing Model & Loading Data..."):
    model, feature_cols = train_model()

if model is None:
    st.error("Could not load data. Please check repository files.")
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
    
    # Create DataFrame and align columns
    df_input = pd.DataFrame([input_data])
    df_final = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        df_final.loc[0, col] = df_input.loc[0, col] if col in df_input.columns else 0
            
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
    # Optional: Show Feature Importance when no prediction is made yet
    st.markdown("---")
    with st.expander("See what drives this model"):
        st.write("Top 5 factors influencing the prediction:")
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(5)
        st.bar_chart(feat_imp, color="#FF4B4B")
