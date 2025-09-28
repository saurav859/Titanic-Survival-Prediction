import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Titanic Survival Prediction')
st.write('Predict the survival probability of a passenger on the Titanic.')

# Load the trained model and feature columns
try:
    model = joblib.load('logistic_regression_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'logistic_regression_model.pkl' and 'feature_columns.pkl' are in the same directory.")
    st.stop()


# Create input widgets for features
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['female', 'male'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('SibSp', 0, 10, 0)
parch = st.slider('Parch', 0, 10, 0)
fare = st.number_input('Fare', value=0.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Preprocess user input
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, feature_columns):
    # Create a dictionary from user inputs
    data = {
        'Pclass': pclass,
        'Sex': 1 if sex == 'male' else 0,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
    }
    input_df = pd.DataFrame([data])

    # Ensure the order of columns matches the training data
    # Add missing Embarked columns if any
    for col in ['Embarked_Q', 'Embarked_S']:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[feature_columns]

    return input_df

if st.button('Predict Survival'):
    # Preprocess the user input
    processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, feature_columns)

    # Make prediction
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    # Display the result
    st.subheader('Prediction Result')
    st.write(f'Predicted Survival Probability: {prediction_proba[0]:.4f}')

    if prediction_proba[0] > 0.5:
        st.success('The model predicts the passenger is likely to survive.')
    else:
        st.error('The model predicts the passenger is not likely to survive.')
        
