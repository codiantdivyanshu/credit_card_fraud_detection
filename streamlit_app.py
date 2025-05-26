import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import joblib 

joblib.dump(model , 'model.pkl')
model = joblib.load("model.pkl")

# Title
st.title("💳 Credit Card Fraud Detection")

# Sidebar input
st.sidebar.header("Input Transaction Details")

def user_input_features():
    input_data = {}
    for i in range(1, 29):
        input_data[f'V{i}'] = st.sidebar.slider(f'V{i}', -20.0, 20.0, 0.0)
    amount = st.sidebar.number_input("Amount", min_value=0.0, value=100.0)
    time = st.sidebar.number_input("Time", min_value=0.0, value=0.0)
    input_data['Amount'] = amount
    input_data['Time'] = time
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Preprocessing
input_df['normAmount'] = StandardScaler().fit_transform(input_df[['Amount']])
input_df['normTime'] = StandardScaler().fit_transform(input_df[['Time']])
input_df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write("🔴 Fraud" if prediction[0] == 1 else "🟢 Legitimate")

st.subheader('Prediction Probability')
st.write(f"Legit: {prediction_proba[0][0]:.2f}, Fraud: {prediction_proba[0][1]:.2f}")
