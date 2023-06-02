import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler


# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Create a LSTM model for exchange rate prediction
def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_model(model, data):
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

# Predict the exchange rate using the trained model
def predict_rate(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_test = np.array([scaled_data[-1]])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_rate = model.predict(x_test)
    predicted_rate = scaler.inverse_transform(predicted_rate)
    return predicted_rate[0][0]



# Create a Keras model for exchange rate prediction
def create_model_keras():
    model_keras = Sequential()
    model_keras.add(Dense(units=50, activation='relu', input_shape=(1,)))
    model_keras.add(Dense(units=50, activation='relu'))
    model_keras.add(Dense(units=1))
    model_keras.compile(optimizer='adam', loss='mean_squared_error')
    return model_keras

# Train the Keras model
def train_model_keras(model, data):
    scaler_keras = MinMaxScaler(feature_range=(0, 1))
    scaled_data_keras = scaler_keras.fit_transform(data_keras)

    x_train_keras = scaled_data_keras[:-1]
    y_train_keras = scaled_data_keras[1:]

    model_keras.fit(x_train_keras, y_train_keras, epochs=20, batch_size=1, verbose=2)

# Predict the exchange rate using the trained model
def predict_rate_keras(model, data):
    scaler_keras = MinMaxScaler(feature_range=(0, 1))
    scaled_data_keras = scaler_keras.fit_transform(data_keras)

    x_test_keras = scaled_data_keras[-1]
    predicted_rate_keras = model_keras.predict(np.array([x_test_keras]))
    predicted_rate_keras = scaler_keras.inverse_transform(predicted_rate_keras)
    return predicted_rate_keras[0][0]

# Main function
def main():
    st.title('Exchange Rate Prediction TensorFlow')
   
    # Upload historical data file
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    
        st.subheader('Historical Data')
        st.dataframe(data)  # Display all loaded values
        
        model = create_model()
        
        st.subheader('Train Model')
        train_model(model, data['Rate'].values.reshape(-1, 1))
        st.write('Model training complete.')
        
        st.subheader('Exchange Rate Prediction')
        prediction = predict_rate(model, data['Rate'].values.reshape(-1, 1))
        st.write('Predicted exchange rate:', prediction)

        st.title('Exchange Rate Prediction Keras')



        st.subheader('Historical Data')
        st.dataframe(data)  # Display all loaded values

        model_keras = create_model_keras()

        st.subheader('Train Model')
        train_model_keras(model_keras, data['Rate'].values.reshape(-1, 1))
        st.write('Model training complete.')

        st.subheader('Exchange Rate Prediction')
        prediction_keras = predict_rate_keras(model_keras, data['Rate'].values.reshape(-1, 1))
        st.write('Predicted exchange rate:', prediction_keras)


# Run the application
if __name__ == '__main__':
    main()
