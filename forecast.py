import streamlit as st
import mxnet as mx
import pandas as pd

# Load historical exchange rate data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df):
    # Normalize the exchange rates
    rates = df['Rate'].values
    min_rate = min(rates)
    max_rate = max(rates)
    normalized_rates = (rates - min_rate) / (max_rate - min_rate)
    df['Normalized Rate'] = normalized_rates

    # Convert the date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    return df

# Build and train the MXNet neural network model
def build_model():
    net = mx.gluon.nn.Sequential()
    net.add(mx.gluon.nn.Dense(10, activation='relu'))
    net.add(mx.gluon.nn.Dense(1))
    net.initialize()
    loss_function = mx.gluon.loss.L2Loss()
    optimizer = mx.gluon.Trainer(net.collect_params(), 'adam')
    return net, loss_function, optimizer

# Train the model using historical data
def train_model(net, loss_function, optimizer, X_train, y_train, epochs=10, batch_size=32):
    dataset = mx.gluon.data.ArrayDataset(X_train, y_train)
    dataloader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        cumulative_loss = 0
        for X, y in dataloader:
            with mx.autograd.record():
                output = net(X)
                loss = loss_function(output, y)
            loss.backward()
            optimizer.step(batch_size)
            cumulative_loss += mx.nd.sum(loss).asscalar()
        epoch_loss = cumulative_loss / len(X_train)
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

# Make predictions using the trained model
def make_predictions(net, X_test):
    predictions = net(X_test)
    return predictions

# Main function
def main():
    st.title("Exchange Rate Prediction")
    
    # File upload
    file = st.file_uploader("Upload CSV file", type="csv")
    
    if file is not None:
        df = load_data(file.name)
        df = preprocess_data(df)
        
        st.subheader("Data")
        st.write(df)
        
        # Prepare training data
        X_train = df['Number'].values.reshape(-1, 1)
        y_train = df['Normalized Rate'].values.reshape(-1, 1)
        
        # Build and train the model
        net, loss_function, optimizer = build_model()
        train_model(net, loss_function, optimizer, X_train, y_train)
        
        # Make predictions
        X_test = X_train
        predictions = make_predictions(net, X_test)
        
        # Denormalize predictions
        min_rate = df['Rate'].min()
        max_rate = df['Rate'].max()
        denormalized_predictions = predictions * (max_rate - min_rate) + min_rate
        
        # Display predictions
        st.subheader("Predictions")
        st.write(pd.DataFrame({'Number': X_test.flatten(), 'Prediction': denormalized_predictions.flatten()}))

if __name__ == '__main__':
    main()
