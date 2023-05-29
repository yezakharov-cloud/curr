import streamlit as st
import pandas as pd
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

# Load historical data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Perform any necessary preprocessing steps
    # e.g., data normalization, feature engineering, etc.
    return data

# Build PyBrain neural network model
def build_model():
    n_inputs = 1  # Number of input features (e.g., historical rates)
    n_hidden = 10  # Number of hidden units in the neural network
    n_outputs = 1  # Number of output units (predicted rate)

    # Create a feed-forward neural network
    network = FeedForwardNetwork()

    # Add layers to the network
    input_layer = LinearLayer(n_inputs)
    hidden_layer = SigmoidLayer(n_hidden)
    output_layer = LinearLayer(n_outputs)

    network.addInputModule(input_layer)
    network.addModule(hidden_layer)
    network.addOutputModule(output_layer)

    # Connect the layers
    input_to_hidden = FullConnection(input_layer, hidden_layer)
    hidden_to_output = FullConnection(hidden_layer, output_layer)

    network.addConnection(input_to_hidden)
    network.addConnection(hidden_to_output)

    # Initialize the network
    network.sortModules()

    return network

# Train the PyBrain neural network model
def train_model(network, data):
    X = data["Number"].values  # Input features
    y = data["Rate"].values  # Target variable (rate)

    # Create a SupervisedDataSet for training
    dataset = SupervisedDataSet(1, 1)
    for i in range(len(X)):
        dataset.addSample(X[i], y[i])

    # Train the network using Backpropagation algorithm
    trainer = BackpropTrainer(network, dataset)
    trainer.train()

# Predict currency rates using the trained model
def predict_rates(network, input_data):
    predictions = []
    for input_value in input_data:
        output_value = network.activate([input_value])
        predictions.append(output_value[0])
    return predictions

# Main Streamlit application
def main():
    st.title("Currency Rate Prediction")
    st.sidebar.title("Settings")

    # Load historical data from CSV file
    file_path = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if file_path is not None:
        data = load_data(file_path)
        data = preprocess_data(data)

        # Build and train the model
        model = build_model()
        train_model(model, data)

        # Get input values for prediction
        input_values = st.sidebar.text_input("Enter input values (comma-separated)")

        if input_values:
            input_data = [float(x.strip()) for x in input_values.split(",")]
            predictions = predict_rates(model, input_data)

            st.subheader("Prediction Results")
            for i in range(len(input_data)):
                st.write(f"Input: {input_data[i]}, Prediction: {predictions[i]}")

if __name__ == "__main__":
    main()
