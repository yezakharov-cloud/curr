import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# Function to load historical data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Function to create an LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train an LSTM model
def train_lstm_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train = []
    y_train = []
    for i in range(1, len(scaled_data)):
        x_train.append(scaled_data[i-1:i])
        y_train.append(scaled_data[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
    return model, scaler

# Function to predict the next day's rate using the LSTM model
def predict_next_day_rate_lstm(model, scaler, data):
    scaled_data = scaler.transform(data)
    
    x_test = np.array([scaled_data[-1]])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_rate = model.predict(x_test)
    predicted_rate = scaler.inverse_transform(predicted_rate)
    return predicted_rate[0][0]

# Function to create a simple neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_shape=(1,)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train a simple neural network model
def train_nn_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train = scaled_data[:-1]
    y_train = scaled_data[1:]

    model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
    return model, scaler

# Function to predict the next day's rate using the simple neural network model
def predict_next_day_rate_nn(model, scaler, data):
    scaled_data = scaler.transform(data)

    x_test = scaled_data[-1]
    predicted_rate = model.predict(np.array([x_test]))
    predicted_rate = scaler.inverse_transform(predicted_rate)
    return predicted_rate[0][0]

# Main function
def main():
    st.title("Gold Price Prediction App")

    # Upload historical gold price data file
    uploaded_gold_file = st.file_uploader('Upload CSV file for historical gold prices', type=['csv'])

    if uploaded_gold_file is not None:
        gold_data = load_data(uploaded_gold_file)

        st.subheader('Historical Gold Price Data')
        st.dataframe(gold_data)

        # Choose the model
        model_choice = st.radio("Choose a model for prediction:", ("LSTM Model", "Simple Neural Network Model"))

        if model_choice == "LSTM Model":
            st.subheader('LSTM Model for Gold Price Prediction')

            lstm_model_file = st.file_uploader('Upload saved LSTM model file', type=['h5'])

            lstm_model = None
            lstm_scaler = None

            if lstm_model_file is not None:
                lstm_model = tf.keras.models.load_model(lstm_model_file)
                lstm_scaler = MinMaxScaler(feature_range=(0, 1))
                lstm_scaler.fit_transform(gold_data)
            
            if lstm_model is None or st.button('Train New LSTM Model'):
                lstm_model, lstm_scaler = train_lstm_model(create_lstm_model(), gold_data)

            next_day_rate_lstm = predict_next_day_rate_lstm(lstm_model, lstm_scaler, gold_data['Price'].values.reshape(-1, 1))

            st.write('Predicted Gold Price for the Next Day (LSTM Model):', next_day_rate_lstm)

        elif model_choice == "Simple Neural Network Model":
            st.subheader('Simple Neural Network Model for Gold Price Prediction')

            nn_model_file = st.file_uploader('Upload saved Neural Network model file', type=['h5'])

            nn_model = None
            nn_scaler = None

            if nn_model_file is not None:
                nn_model = tf.keras.models.load_model(nn_model_file)
                nn_scaler = MinMaxScaler(feature_range=(0, 1))
                nn_scaler.fit_transform(gold_data)
            
            if nn_model is None or st.button('Train New Neural Network Model'):
                nn_model, nn_scaler = train_nn_model(create_nn_model(), gold_data)

            next_day_rate_nn = predict_next_day_rate_nn(nn_model, nn_scaler, gold_data['Price'].values.reshape(-1, 1))

            st.write('Predicted Gold Price for the Next Day (Neural Network Model):', next_day_rate_nn)

    # Save the models
    if lstm_model is not None:
        lstm_model.save("lstm_model.h5")

    if nn_model is not None:
        nn_model.save("nn_model.h5")

if __name__ == '__main__':
    main()
