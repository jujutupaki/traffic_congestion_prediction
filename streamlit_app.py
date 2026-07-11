import streamlit as st
import pandas as pd

st.title('🚗 Traffic Congestion Prediction')

st.write('Description!')

dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Traffic_Dataset.csv')
dataset_df
      
