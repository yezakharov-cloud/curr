import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the historical exchange rate data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    st.write("Last 3 records of uploaded data:")
    st.write(data.tail(3))
    return data

# Preprocess the data
def preprocess_data(data):
    # Convert dates to numerical values
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.to_julian_date)
    
    # Scale the exchange rate values to a range between 0 and 1
    scaler = MinMaxScaler()
    data['Exchange Rate'] = scaler.fit_transform(data['Exchange Rate'].values.reshape(-1, 1))
    
    return data

# Build the neural network model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions using the trained model
def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions

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
        X = data[['Sequence Number', 'Date']].values
        y = data['Exchange Rate'].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build the neural network model
        input_shape = (X_train.shape[1],)
        model = build_model(input_shape)
        
        # Train the model
        train_model(model, X_train, y_train)
        
        # Make predictions
        predictions = predict(model, X_test)
        
        # Display predicted exchange rates
        st.write("Predicted Exchange Rates:")
        st.write(predictions)
    else:
        st.sidebar.write("Please upload a CSV file.")

# Run the app
if __name__ == '__main__':
    main()
