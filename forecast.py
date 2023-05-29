import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Create a PyTorch model for exchange rate prediction
class ExchangeRateModel(nn.Module):
    def __init__(self):
        super(ExchangeRateModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        output, _ = self.lstm1(x)
        output, _ = self.lstm2(output[:, -1:, :])
        output = self.fc(output[:, -1, :])
        return output

# Train the PyTorch model
def train_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train = []
    y_train = []
    for i in range(1, len(scaled_data)):
        x_train.append(scaled_data[i-1:i])
        y_train.append(scaled_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict the exchange rate using the trained PyTorch model
def predict_rate(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_test = torch.from_numpy(scaled_data[-1:]).float()

    with torch.no_grad():
        prediction = model(x_test)

    prediction = scaler.inverse_transform(prediction.numpy())
    return prediction[0][0]

# Main function
def main():
    st.title('Exchange Rate Prediction')

    # Upload historical data file
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.subheader('Historical Data')
        st.dataframe(data)  # Display all loaded values

        model = ExchangeRateModel()

        st.subheader('Train Model')
        train_model(model, data['Rate'].values.reshape(-1, 1))
        st.write('Model training complete.')

        st.subheader('Exchange Rate Prediction')
        prediction = predict_rate(model, data['Rate'].values.reshape(-1, 1))
        st.write('Predicted exchange rate:', prediction)

# Run the application
if __name__ == '__main__':
    main()
