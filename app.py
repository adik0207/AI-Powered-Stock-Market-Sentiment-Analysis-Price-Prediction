import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_stock_model.h5')

# Load the scalers
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Define user-friendly feature names
feature_names = [
    "Average Price (Last 50 Days)", "Average Price (Last 200 Days)", "Recent Trend (Last 50 Days)", "Long-Term Trend (Last 200 Days)", 
    "Buying Strength (0-100)", "Price Momentum", "Momentum Confirmation", "High Expected Price Range", "Low Expected Price Range", 
    "Buying & Selling Volume", "Price Strength (0-100)", "Number of Shares Traded"
]

# Function to make predictions
def predict_price(input_data):
    input_scaled = scaler_X.transform(input_data)  # Normalize input
    input_scaled = np.reshape(input_scaled, (1, input_scaled.shape[0], input_scaled.shape[1]))  # Reshape for LSTM
    prediction = model.predict(input_scaled)
    return scaler_y.inverse_transform(prediction)[0][0]

# Streamlit UI
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 AI Powered Stock Price Predictor")
st.write("Enter the latest stock indicators to predict the next day's price.")

# User input for each stock feature
feature_values = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"{feature}", value=0.0, key=f"feature_{i}")
    feature_values.append(value)

# Predict Button
if st.button("Predict Stock Price"):
    input_array = np.array(feature_values).reshape(1, -1)
    predicted_price = predict_price(input_array)
    st.success(f"Predicted Stock Price: ₹ {predicted_price:.2f}")

#Fascinating fact section
st.subheader("💡 Did You Know?")
st.write("The world's first stock exchange was established in **Amsterdam in 1602** by the Dutch East India Company. It revolutionized the way businesses raised capital and laid the foundation for modern financial markets!")

# Footer
st.markdown("---")
st.markdown("🔹 *Stock market predictions are based on past trends and may not always be accurate. Invest wisely!* 📊")

# Signature
st.markdown("**Developed by Aditya Khare, 2025**")
