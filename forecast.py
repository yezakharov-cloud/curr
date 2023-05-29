import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from cntk import input_variable, relu, training_session, Trainer, learning_rate_schedule, UnitType
from cntk.layers import Dense
from cntk.ops import squared_error

# Load the historical exchange rate data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Convert dates to numerical values
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.to_julian_date)
    
    # Scale the exchange rate values to a range between 0 and 1
    scaler = MinMaxScaler()
    data['Rate'] = scaler.fit_transform(data['Rate'].values.reshape(-1, 1))
    
    return data

# Build the neural network model using Microsoft CNTK
def build_model(input_dim):
    input_var = input_variable(input_dim)
    hidden_layer = Dense(64, activation=relu)(input_var)
    output_layer = Dense(1)(hidden_layer)
    return output_layer

# Train the model
def train_model(model, X_train, y_train):
    learning_rate = learning_rate_schedule(0.1, UnitType.minibatch)
    loss = squared_error(model, y_train)
    learner = Trainer(model, loss, learning_rate)
    batch_size = 32
    num_epochs = 50
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            learner.train_minibatch({model.arguments[0]: x_batch, loss.arguments[0]: y_batch})
    return model

# Make predictions using the trained model
def predict(model, X_test):
    return model.eval({model.arguments[0]: X_test})

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
        X = data[['Number', 'Date']].values
        y = data['Rate'].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build the neural network model
        input_dim = X_train.shape[1]
        model = build_model(input_dim)
        
        # Train the model
        model = train_model(model, X_train, y_train)
        
        # Make predictions
        predictions = predict(model, X_test)
        
        # Inverse scale the predictions
        scaler = MinMaxScaler()
        scaler.fit(data['Rate'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)
        
        # Display predicted exchange rates
        st.write("Predicted Exchange Rates:")
        st.write(predictions)
    else:
        st.sidebar.write("Please upload a CSV file.")

#
