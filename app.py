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
