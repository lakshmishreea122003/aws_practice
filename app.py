import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("C:/Users/Lakshmi/Downloads/house_price_model.pkl", 'rb'))

st.title("House Price Prediction")

# Input all 13 features of the Boston dataset
CRIM = st.number_input("CRIM: Per capita crime rate")
ZN = st.number_input("ZN: Proportion of residential land zoned")
INDUS = st.number_input("INDUS: Proportion of non-retail business acres")
CHAS = st.number_input("CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
NOX = st.number_input("NOX: Nitric oxide concentration (parts per 10 million)")
RM = st.number_input("RM: Average number of rooms per dwelling")
AGE = st.number_input("AGE: Proportion of owner-occupied units built prior to 1940")
DIS = st.number_input("DIS: Weighted distances to five Boston employment centers")
RAD = st.number_input("RAD: Index of accessibility to radial highways")
TAX = st.number_input("TAX: Full-value property tax rate per $10,000")
PTRATIO = st.number_input("PTRATIO: Pupil-teacher ratio by town")
B = st.number_input("B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents")
LSTAT = st.number_input("LSTAT: Percentage of lower status population")

# Make predictions
if st.button("Predict"):
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    prediction = model.predict(features)
    st.success(f"Predicted House Price: {prediction[0][0]:.2f}")


