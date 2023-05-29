import streamlit as st
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, Trainer

# Define the neural network model using MXNet
class ExchangeRatePredictor(nn.Block):
    def __init__(self):
        super(ExchangeRatePredictor, self).__init__()
        self.fc = nn.Dense(10, activation='relu')
        self.output = nn.Dense(1)

    def forward(self, x):
        x = self.fc(x)
        return self.output(x)

# Load historical exchange rate data from a CSV file
def load_data(file_path):
    data = mx.ndarray.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, :-1], data[:, -1]

# Preprocess the data for training the neural network
def preprocess_data(data):
    # Normalize the data
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std
    return data, mean, std

# Train the neural network
def train_model(data, labels):
    model = ExchangeRatePredictor()
    criterion = gluon.loss.L2Loss()
    trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

    for epoch in range(1000):
        with autograd.record():
            output = model(data)
            loss = criterion(output, labels)
        loss.backward()
        trainer.step(data.shape[0])

    return model

# Predict the exchange rate using the trained model
def predict_rate(model, data, mean, std):
    data = (data - mean) / std
    output = model(data)
    return output * std[-1] + mean[-1]

# Streamlit application
def main():
    st.title('Exchange Rate Prediction')
    st.write('Upload the historical exchange rate data file (CSV format):')
    file = st.file_uploader('Choose a CSV file', type='csv')

    if file is not None:
        data, labels = load_data(file.name)
        data, mean, std = preprocess_data(data)
        model = train_model(data, labels)
        last_data = nd.array(data[-1])
        prediction = predict_rate(model, last_data, mean, std)
        st.write(f'The predicted exchange rate for the next time step is: {prediction.asscalar()}')

if __name__ == '__main__':
    main()
