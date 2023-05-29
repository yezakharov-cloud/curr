import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data


# Define the neural network model using PyTorch
class ExchangeRatePredictor(nn.Module):
    def __init__(self):
        super(ExchangeRatePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Preprocess the data for training the neural network
def preprocess_data(data):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data['Rate'] = scaler.fit_transform(data['Rate'].values.reshape(-1, 1))
    return data, scaler


# Train the neural network
def train_model(data):
    model = ExchangeRatePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.Tensor(data['Number'].values).unsqueeze(1)
    y = torch.Tensor(data['Rate'].values).unsqueeze(1)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# Predict the exchange rate using the trained model
def predict_rate(model, rate, scaler):
    x = torch.Tensor([rate])
    x = scaler.transform(x.reshape(-1, 1))
    x = torch.Tensor(x)
    prediction = model(x)
    prediction = scaler.inverse_transform(prediction.detach().numpy().reshape(-1, 1))
    return prediction.item()




# Streamlit application
def main():
    st.title('Exchange Rate Prediction')
    st.write('Upload the historical exchange rate data file (CSV format):')
    file = st.file_uploader('Upload CSV file', type=['csv'])

    if file is not None:
        data = load_data(file)
        data, scaler = preprocess_data(data)
        model = train_model(data)
        last_rate = data['Rate'].values[-1]
        prediction = predict_rate(model, last_rate, scaler)
        st.write(f'The predicted exchange rate for the next time step is: {prediction}')


if __name__ == '__main__':
    main()
