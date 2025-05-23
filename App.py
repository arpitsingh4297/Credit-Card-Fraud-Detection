import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Streamlit app title and description
st.title("Credit Card Fraud Detection")
st.markdown("""
This app uses a trained KNN model (F1-score: 0.82) to predict credit card fraud in real-time. 
Upload a CSV file containing transaction data with columns: `Amount`, `Time`, `V17`, `V14`, `V12`, `V3`, `V10`, `V7`, `V16`, `V11`.
The model will predict whether each transaction is fraudulent (1) or non-fraudulent (0).
""")

# File uploader
uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")

# Load model
model_path = 'fraud_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure 'fraud_model.pkl' is in the same directory as this app.")
else:
    model = joblib.load(model_path)
    st.success("KNN model loaded successfully.")

    if uploaded_file is not None:
        try:
            # Read CSV
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", data.head())

            # Validate required columns
            required_columns = ['Amount', 'Time', 'V17', 'V14', 'V12', 'V3', 'V10', 'V7', 'V16', 'V11']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Preprocess data
                scaler = StandardScaler()
                data['Scaled_Amount'] = scaler.fit_transform(data[['Amount']])
                data['Scaled_Time'] = scaler.fit_transform(data[['Time']])
                
                # Select features
                X = data[['V17', 'V14', 'V12', 'V3', 'V10', 'V7', 'V16', 'V11']]
                
                # Make predictions
                predictions = model.predict(X)
                data['Prediction'] = predictions
                data['Prediction_Label'] = data['Prediction'].map({1: 'Fraud', 0: 'Non-Fraud'})
                
                # Display results
                st.subheader("Prediction Results")
                result_df = data[['Amount', 'Time', 'V17', 'V14', 'V12', 'V3', 'V10', 'V7', 'V16', 'V11', 'Prediction_Label']]
                st.dataframe(result_df)
                
                # Summary
                fraud_count = (predictions == 1).sum()
                total_count = len(predictions)
                st.write(f"**Summary**: {fraud_count} out of {total_count} transactions predicted as fraudulent ({fraud_count/total_count*100:.2f}%).")
                
                # Download predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
                
                # Business insights
                st.subheader("Business Insights")
                st.markdown("""
                - **Focus on transactions with low V14 and V17 values**, as these are strong indicators of fraud.
                - Deploy this KNN model for real-time fraud detection to minimize financial losses.
                - Monitor Cluster 2 (higher fraud proportion: 0.57%) for risky transaction patterns.
                """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Instructions
st.subheader("How to Use")
st.markdown("""
1. Ensure the CSV file contains the required columns: `Amount`, `Time`, `V17`, `V14`, `V12`, `V3`, `V10`, `V7`, `V16`, `V11`.
2. Upload the CSV using the file uploader.
3. View the prediction results in the table (1: Fraud, 0: Non-Fraud).
4. Download the predictions as a CSV for further analysis (e.g., in PowerBI).
""")