import streamlit as st
import pandas as pd
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from datetime import datetime

# Load historical exchange rate data from a CSV file
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Prepare data for training the neural network
def prepare_data(data):
    ds = SupervisedDataSet(1, 1)
    for i in range(len(data)):
        ds.addSample(data['Number'][i], data['Rate'][i])
    return ds

# Train the neural network
def train_network(ds):
    net = FeedForwardNetwork()
    in_layer = LinearLayer(1)
    hidden_layer = SigmoidLayer(3)
    out_layer = LinearLayer(1)
    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)
    in_to_hidden = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)
    net.sortModules()
    trainer = BackpropTrainer(net, ds)
    trainer.trainEpochs(1000)
    return net

# Predict exchange rate using the trained network
def predict_rate(net, input_data):
    output = net.activate(input_data)
    return output

# Main function
def main():
    st.title("Exchange Rate Prediction")

    # Load historical data from CSV
    file_path = st.file_uploader("Upload a CSV file", type="csv")
    if file_path is not None:
        data = load_data(file_path)
        st.write("Historical Data:")
        st.write(data)

        # Prepare data for training
        ds = prepare_data(data)

        # Train the neural network
        net = train_network(ds)

        # Prediction
        st.write("Prediction:")
        number = st.number_input("Enter a number:")
        prediction = predict_rate(net, number)
        st.write(f"Predicted rate: {prediction[0]:.2f}")
    else:
        st.write("Please upload a CSV file.")

if __name__ == '__main__':
    main()
