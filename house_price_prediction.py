import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict price")

# Sample dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [1, 2, 2, 3, 3, 4],
    'Price': [40000, 50000, 60000, 75000, 90000, 110000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
area = st.number_input("Area (in sq ft)", min_value=500, max_value=3000, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms]])
    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")
