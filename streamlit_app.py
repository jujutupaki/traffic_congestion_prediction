import streamlit as st
import pandas as pd

st.title('🚗 Traffic Congestion Prediction')

st.info("""Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: \n
"Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".

What you can explore here:\n
• **Dataset Overview:** The final traffic-weather data using selected features\n
• **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
• **Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions""")

with st.expander('View Final Traffic-Weather Dataset'):
      dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Selected_Traffic_Data.csv')
      dataset_df

# User-defined features
with st.sidebar:
      st.header("Input features below to generate a prediction:")

#CV SPlit
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
X = train_val.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y = train_val['Simulated Traffic Level']
X_test = test.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y_test = test['Simulated Traffic Level']

with st.expander('Hello'):
      X
      y
with st. expander('World'):
      X_test
      y_test
