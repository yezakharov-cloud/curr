import streamlit as st
import pandas as pd
import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from sklearn.preprocessing import MinMaxScaler

# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Create a PyBrain model for exchange rate prediction
def create_model():
    model = FeedForwardNetwork()
    
    input_layer = LinearLayer(1)
    hidden_layer = SigmoidLayer(50)
    output_layer = LinearLayer(1)
    
    model.addInputModule(input_layer)
    model.addModule(hidden_layer)
    model.addOutputModule(output_layer)
    
    input_to_hidden = FullConnection(input_layer, hidden_layer)
    hidden_to_output = FullConnection(hidden_layer, output_layer)
    
    model.addConnection(input_to_hidden)
    model.addConnection(hidden_to_output)
    
    model.sortModules()
    
    return model

# Train the PyBrain model
def train_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train = []
    y_train = []
    for i in range(1, len(scaled_data)):
        x_train.append(scaled_data[i-1:i])
        y_train.append(scaled_data[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    
    for epoch in range(20):
        for i in range(len(x_train)):
            model.activate(x_train[i])
            model.backActivate(y_train[i])
    
# Predict the exchange rate using the trained model
def predict_rate(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_test = np.array([scaled_data[-1]])
    x_test = x_test.reshape(x_test.shape[0], 1)
    
    predicted_rate = model.activate(x_test)
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
