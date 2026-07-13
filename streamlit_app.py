import streamlit as st
import pandas as pd

st.title('🚗 Traffic Congestion Prediction')

st.info('Welcome to \033[1mbrr-traffic.streamlit.app\033[0m, the interactive dashboard for our thesis: "Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".\n
What you can explore here:\n
🚦 Dataset Overview: The final traffic-weather data using selected features\n
📈 Model Evaluation Dashboard: Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
🔮 Interactive Traffic Predictor: Input custom weather conditions, dates, and times to generate real-time congestion predictions')

File "/mount/src/traffic_congestion_prediction/streamlit_app.py", line 6
  st.write('Welcome to \033[1mbrr-traffic.streamlit.app\033[0m, the interactive dashboard for our thesis: "Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".\n
           ^
SyntaxError: unterminated string literal (detected at line 6)

with st.expander('View Final Traffic-Weather Dataset'):
      dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Selected_Features_Traffic_Dataset.csv')
      dataset_df
