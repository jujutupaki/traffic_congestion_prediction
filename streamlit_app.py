import streamlit as st
import pandas as pd

st.title('🚗 Traffic Congestion Prediction')

st.writest.write("""
Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: "Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches"

What you can explore here:
🚦 **Dataset Overview:** The final traffic-weather data using selected features
📈 **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models
🔮 **Interactive Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions.
""")

with st.expander('View Final Traffic-Weather Dataset'):
      dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Selected_Features_Traffic_Dataset.csv')
      dataset_df
