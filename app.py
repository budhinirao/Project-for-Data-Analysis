# app.py
import streamlit as st
import pandas as pd
from ml import train_extra_trees

# Load data
@st.cache
def load_data():
    return pd.read_csv('Housing.csv')

# Main function
def main():
    st.title("Boston Housing Price Prediction App")

    # Load data
    data = load_data()

    # Sidebar
    st.sidebar.header('Input Features')
    selected_features = st.sidebar.multiselect('Select features:', data.columns.tolist(), default=['CRIM', 'NOX', 'RM', 'AGE'])

    # Select data
    X = data[selected_features]
    y = data['MEDV']

    # Train the model
    model, mse, cv_score, r2 = train_extra_trees(X, y)

    # Display model evaluation results
    st.write('*Model Evaluation Results*')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Cross-Validation Score (CV Score): {cv_score}')
    st.write(f'R^2 Score: {r2}')

if _name_ == "_main_":
    main()