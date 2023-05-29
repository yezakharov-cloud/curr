import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import theano
import theano.tensor as T
from theano import function
from theano import shared
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

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

# Build the neural network model using Theano
def build_model(input_dim):
    W_xh = shared(np.random.randn(input_dim, 64) * 0.01)
    W_hh = shared(np.random.randn(64, 64) * 0.01)
    W_hy = shared(np.random.randn(64, 1) * 0.01)
    b_h = shared(np.zeros(64))
    b_y = shared(0.)
    
    X = T.matrix()
    y = T.vector()
    
    def step(x_t, h_tm1):
        h_t = sigmoid(T.dot(x_t, W_xh) + T.dot(h_tm1, W_hh) + b_h)
        y_t = T.dot(h_t, W_hy) + b_y
        return h_t, y_t
    
    [h, y_pred], _ = theano.scan(
        step,
        sequences=X,
        outputs_info=[dict(initial=T.zeros(64)), None]
    )
    
    cost = T.mean((y - y_pred.flatten())**2)
    gradients = T.grad(cost, [W_xh, W_hh, W_hy, b_h, b_y])
    
    learning_rate = 0.1
    updates = [
        (param, param - learning_rate * gradient)
        for param, gradient in zip([W_xh, W_hh, W_hy, b_h, b_y], gradients)
    ]
    
    train_model = function(
        inputs=[X, y],
        outputs=cost,
        updates=updates
    )
    
    predict_model = function(
        inputs=[X],
        outputs=y_pred.flatten()
    )
    
    return train_model, predict_model

# Train the model
def train_model(train_fn, X_train, y_train):
    num_epochs = 50
    for epoch in range(num_epochs):
        cost = train_fn(X_train, y_train)
        if epoch % 10 == 0:
            print("Epoch {}: Cost = {}".format(epoch, cost))

# Make predictions using the trained model
def predict(predict_fn, X_test):
    return predict_fn(X_test)

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
        
        #
