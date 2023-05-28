import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Define the PyTorch dataset
class ExchangeRateDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define the PyTorch neural network model
class ExchangeRateModel(nn.Module):
    def __init__(self):
        super(ExchangeRateModel, self).__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load historical exchange rate data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Create a PyTorch model for exchange rate prediction
def create_model():
    model = ExchangeRateModel()
    return model

# Train the PyTorch model
def train_model(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    dataset = ExchangeRateDataset(scaled_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        running_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()

            inputs = inputs.unsqueeze(2)
            outputs = model(inputs)

            loss = criterion(outputs, inputs[:, -1, :])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        st.write(f'Epoch {epoch + 1} Loss: {running_loss / len(dataloader)}')

# Predict the exchange rate using the trained model
def predict_rate(model, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    inputs = torch.from_numpy(scaled_data).unsqueeze(0).unsqueeze(2).float()
    prediction = model(inputs)
    prediction = scaler.inverse_transform(prediction.detach().numpy())[0][0]
    return prediction

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
