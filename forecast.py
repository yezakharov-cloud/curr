
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.write(f"Веб-сервіс формування рекомендацій для коригування прогнозу фінансово-економічних показників на основі аналізу новин")

# Display the radio buttons
choice = st.radio("Оберіть опцію:", ("Курс валюти не відомий", "Курс валюти відомий"))

if choice == "Курс валюти не відомий":
    current_rate = 0
elif choice == "Курс валюти відомий":
    st.title("Введіть відомий курс валюти")
    current_rate = st.number_input('приклад:(50)')
    if current_rate:
        st.write(f"Поточний курс: {current_rate}")

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
def create_model2():
    model2 = Sequential()
    model2.add(Dense(units=50, activation='relu', input_shape=(1,)))
    model2.add(Dense(units=50, activation='relu'))
    model2.add(Dense(units=1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    return model2

# Train the Keras model
def train_model2(model2, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train = scaled_data[:-1]
    y_train = scaled_data[1:]

    model2.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

# Predict the exchange rate using the trained model
def predict_rate2(model2, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_test = scaled_data[-1]
    predicted_rate = model2.predict(np.array([x_test]))
    predicted_rate = scaler.inverse_transform(predicted_rate)
    return predicted_rate[0][0]

# Main function
def main():
   
    # Upload historical data file
    uploaded_file = st.file_uploader('Завантажте файл CSV', type=['csv'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.title('Прогноз обмінного курсу Модель 1')

        st.subheader('Історичні дані')
        st.dataframe(data)  # Display all loaded values
        
        model = create_model()
        
        st.subheader('Навчання моделі Модель 1 ...')
        train_model(model, data['Rate'].values.reshape(-1, 1))
        st.write('Навчання моделі Модель 1 завершено')
        
        st.subheader('Прогноз обмінного курсу Модель 1')
        prediction = predict_rate(model, data['Rate'].values.reshape(-1, 1))

        st.write('Прогнозований обмінний курс:', prediction)

        st.title('Прогнозування обмінного курсу Модель 2')

        st.subheader('Історичні дані')
        st.dataframe(data)  # Display all loaded values

        model2 = create_model2()

        st.subheader('Навчання моделі Модель 2 ...')
        train_model2(model2, data['Rate'].values.reshape(-1, 1))
        st.write('Навчання моделі Модель 2 завершено')

        st.subheader('Прогнозування обмінного курсу Модель 2')
        prediction2 = predict_rate2(model2, data['Rate'].values.reshape(-1, 1))

        st.write('Прогнозований обмінний курс:', prediction2)
    
                
        if current_rate != 0:

            error1 = abs((current_rate - prediction) / current_rate) * 100
            error2 = abs((current_rate - prediction2) / current_rate) * 100

            st.title("Прогнозування обмінного курсу")
            st.write(f"Поточний курс: {current_rate}")
            st.write(f"Прогнозування Модель 1: ", round(prediction,4))
            st.write(f"Прогнозування Модель 2: ", round(prediction2,4))
            st.title("Похибка прогнозування обмінного курсу")
            st.write(f"Похибка Модель 1: ", round(error1,2),"%")
            st.write(f"Похибка Модель 2: ", round(error2,2),"%")




    # Run the application
if __name__ == '__main__':
    main()
