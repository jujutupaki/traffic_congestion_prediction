import streamlit as st
import pandas as pd
import datetime

st.title('🚗 Traffic Congestion Prediction')

st.info("""Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: \n
"Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".

What you can explore here:\n
🔴 **Dataset Overview:** The final traffic-weather dataset using selected features\n
🟡 **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
🟢 **Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions""")

with st.expander('View Final Traffic-Weather Dataset'):
      dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Selected_Traffic_Data.csv')
      dataset_df

# User-defined features
with st.sidebar:
      st.header("Input features below to generate a prediction:")
      date = st.datetime_input(
      "1. Select date and time:",
      datetime.datetime(2025, 11, 19, 16, 45),
      )
      #get hour, minute, and day of year
      temp = st.slider("2. Select temperature (°C):", 0.0, 30.0, 17.5)
      soil_temp_0 = st.slider("3. Select soil temperature (0-7 cm):", 0.0, 30.0, 18.6)
      driving_direction = st.slider("4. Select driving direction: 0 (Backward), 1 (Forward)", 0, 1, 0)
      app_temp = st.slider("5. Select apparent temperature (°C):", 0.0, 27.0, 18.8)
      soil_temp_7 = st.slider("6. Select soil temperature (7-28 cm):", 0.0, 30.0, 18.9)
      s_pressure = st.slider("7. Select surface pressure:", 800.0, 860.0, 846.5)
      v_pressure = st.slider("8. Select vapour pressure:", 0.0, 2.0, 1.88)

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚗",
    initial_sidebar_state="collapsed",
)
      
#CV SPlit
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
X = train_val.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y = train_val['Simulated Traffic Level']
X_test = test.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y_test = test['Simulated Traffic Level']

