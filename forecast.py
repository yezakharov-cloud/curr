import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Create a Keras model for exchange rate prediction
def create_model():
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_shape=(1,)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the Keras model
def train_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train = scaled_data[:-1]
    y_train = scaled_data[1:]

    model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

# Predict the exchange rate using the trained model
def predict_rate(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_test = scaled_data[-1]
    predicted_rate = model.predict(np.array([x_test]))
    predicted_rate = scaler.inverse_transform(predicted_rate)
    return predicted_rate[0][0]

# Main function
def main():
    st.title('Exchange Rate Prediction')

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

# Run the application
if __name__ == '__main__':
    main()
