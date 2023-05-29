import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers

# Load the historical exchange rate data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Scale the exchange rate values to a range between 0 and 1
    scaler = MinMaxScaler()
    data['Rate'] = scaler.fit_transform(data['Rate'].values.reshape(-1, 1))

    # Convert the date to numerical representation
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = (data['Date'] - data['Date'].min()).dt.days

    return data

# Build the neural network model using Keras
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(1)
    ])
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions using the trained model
def predict(model, X_test):
    return model.predict(X_test).flatten()

# Main function
def main():
    # Set Streamlit app title and layout
    st.title("Exchange Rate Prediction")
    st.sidebar.title("Options")

    # Load and preprocess the data
    file_path = st.sidebar.file_uploader("Upload CSV file", type="csv")

    if file_path is not None:
        data = load_data(file_path)
        data = preprocess_data(data)

        # Split the data into input features (X) and target variable (y)
        X = data[['Number',
