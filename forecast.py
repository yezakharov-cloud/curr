import streamlit as st
import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Function to load historical exchange rate data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    st.write("Last 3 records of uploaded data:")
    st.write(data.tail(3))
    return data

# Function to create a TensorFlow model
def create_tensorflow_model():
    # Implement your TensorFlow model creation code here
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

# Function to train a TensorFlow model
def train_tensorflow_model(model, data, end_date):
    # Split the data into features and target
    x = data.loc[data['Date'] <= end_date, ['Sequence Number']]
    y = data.loc[data['Date'] <= end_date, ['Exchange Rate']]

    # Normalize the data
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=10, batch_size=32)
    st.write("Training completed!")

# Function to create a PyTorch model
def create_pytorch_model():
    # Implement your PyTorch model creation code here
    model = torch.nn.Sequential()
    # Add layers to the model
    return model

# Function to train a PyTorch model
def train_pytorch_model(model, data):
    # Implement your PyTorch model training code here
    st.write("Training PyTorch model...")

# Function to predict exchange rate using TensorFlow model
def predict_tensorflow_rate(date):
    # Normalize the input date
    normalized_date = (date - data['Sequence Number'].mean()) / data['Sequence Number'].std()
    # Perform prediction
    predicted_rate = model.predict(normalized_date)
    st.write("Predicted exchange rate using TensorFlow:", predicted_rate)

# Function to predict exchange rate using PyTorch model
def predict_pytorch_rate(date):
    # Implement your PyTorch prediction code here
    st.write("Predicting exchange rate using PyTorch...")

# Main function
def main():
    st.title("Exchange Rate Prediction")

    # CSV file upload
    st.header("Upload Historical Data (CSV)")
    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        data = load_data(file)
        
     # Button to create TensorFlow model
    if st.button("Create TensorFlow Model"):
        tensorflow_model = create_tensorflow_model()
        

    # TensorFlow model training button
    if st.button("Train a TensorFlow model"):
        tensorflow_model = create_tensorflow_model()
        train_tensorflow_model(tensorflow_model, data)

    # PyTorch model training button
    if st.button("Train a PyTorch model"):
        pytorch_model = create_pytorch_model()
        train_pytorch_model(pytorch_model, data)

    # Date selection for prediction
    st.header("Select Date for Exchange Rate Prediction")
    selected_date = st.date_input("Select a date")

    # Predict exchange rate using TensorFlow button
    if st.button("Currency rate prediction based on the TensorFlow model"):
        predict_tensorflow_rate(selected_date)

    # Predict exchange rate using PyTorch button
    if st.button("Currency rate prediction based on the PyTorch model"):
        predict_pytorch_rate(selected_date)

if __name__ == "__main__":
    main()
