import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import mxnet as mx
from mxnet import gluon, autograd, ndarray
from mxnet.gluon import nn, Trainer
from mxnet.gluon.loss import L2Loss

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

# Build the neural network model using MXNet
def build_model(input_dim):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(1))
    return net

# Train the model
def train_model(model, X_train, y_train):
    loss = L2Loss()
    trainer = Trainer(model.collect_params(), 'adam')
    num_epochs = 50
    batch_size = 32
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        cumulative_loss = 0.0
        for X, y in train_data:
            with autograd.record():
                output = model(X)
                l = loss(output, y)
            l.backward()
            trainer.step(batch_size)
            cumulative_loss += ndarray.mean(l).asscalar()
        if epoch % 10 == 0:
            print("Epoch {}: Loss = {}".format(epoch, cumulative_loss / len(train_data)))

# Make predictions using the trained model
def predict(model, X_test):
    return model(X_test).asnumpy().flatten()

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

        # Convert the data to MXNet NDArray format
        X_train = ndarray.array(X_train)
        y_train = ndarray.array(y_train)

        # Build the neural network model
        input_dim = X_train.shape[1]
        model = build_model(input_dim)
        model.initialize(mx.init.Xavier())

        # Train the model
        train_model(model, X_train, y_train)

        # Convert the test data to MXNet NDArray format
        X_test = ndarray.array(X_test)

        # Make predictions
        predictions = predict(model, X_test)

        #
