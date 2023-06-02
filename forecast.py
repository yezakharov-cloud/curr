import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Function to create a TensorFlow model
def create_tensorflow_model():
     st.write('TensorFlow works OK')

    
# Function to create a Keras model
def create_keras_model():
    st.write('Keras works OK')

# Main function
def main():
    st.title("веб-сервіс формування рекомендацій для коригування прогнозу фінансово-економічних показників на основі аналізу новин")


if __name__ == "__main__":
    main()
