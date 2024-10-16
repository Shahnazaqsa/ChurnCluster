import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Load pre-trained model used in Churn prediction
model = pickle.load(open('churn_model.sav', 'rb'))

# app title
st.title('Customer Churn Prediction and customer Segmentation')

# File uploader to accept CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    input_data = pd.read_csv(uploaded_file)

    st.success("CSV file successfully loaded!")
    
    # Display the uploaded data 
    st.write("Here's a preview of original data:")
    st.dataframe(input_data.head())  # Show first few rows of the file
    
    # Process Data 
    input_data.TotalCharges = pd.to_numeric(input_data.TotalCharges, errors='coerce')
    input_data.dropna(how='any', inplace=True)
    
    # Create tenure groups
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    input_data['tenure_group'] = pd.cut(input_data.tenure, range(1, 80, 12), right=False, labels=labels)
    
    # Deleting unnecessary columns
    input_data.drop(columns=['tenure', 'customerID'], axis=1, inplace=True)
    
    # Convert the 'Churn' column to binary (1: Yes, 0: No)
    if 'Churn' in input_data.columns:
        input_data['Churn'] = np.where(input_data.Churn == 'Yes', 1, 0)
        input_data = input_data.drop(columns=['Churn']) 
    
    # Display the processed data
    st.write("Here's a preview of processed data:")
    st.dataframe(input_data.head())

    # Ensure the uploaded CSV has the required columns (it is just for my personal understanding)
    required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
                        'TotalCharges', 'tenure_group']
    
    if all(col in input_data.columns for col in required_columns):
        st.success('All required columns are present!')

        # Button to trigger the prediction
        if st.button('Predict'):
            # Preprocess input data performing one-hot encoding 
            input_data = pd.get_dummies(input_data)
            # Ensure the columns are in the correct order/format expected by the model
            # This might involve aligning columns to the same order/columns used in the training data
            model_columns = model.feature_names_in_  # Assuming the model has this attribute
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            # Make predictions for each row in the DataFrame
            predictions = model.predict(input_data)
            
            # Get the prediction probabilities
            prediction_probs = model.predict_proba(input_data)

            # Add the predictions and probabilities to the DataFrame
            input_data['Churn_Prediction'] = predictions
            input_data['Churn_Probability (%)'] = prediction_probs[:, 1] * 100  # Convert to percentage

            # Display the predictions with probabilities
            st.write("Predictions for the uploaded data (with probabilities):")
            st.dataframe(input_data[['Churn_Prediction', 'Churn_Probability (%)']])  # Show predictions and probabilities
            # Clusterin the customers
            # deleting the target column
            if 'Churn' in input_data.columns:
                input_data.drop(columns=['Churn'], axis=1, inplace=True)
            scaler = StandardScaler()
            scaler_data = scaler.fit_transform(input_data)
            input_data['Cluster'] = np.random.choice([0, 1, 2], size=len(input_data))  # Random clusters for example
            # Plot the cluster count using sns.countplot
            st.write("Cluster distribution:")
            plt.figure(figsize=(8, 4))
            sns.countplot(x='Cluster', data=input_data)
            st.pyplot(plt)
    else:
        st.error('The CSV file does not have all the required columns. Please upload a valid file.')
