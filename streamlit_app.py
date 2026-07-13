import streamlit as st
import pandas as pd
import datetime

st.title('🚗 Traffic Congestion Prediction')

st.info("""Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: \n
"Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".

What you can explore here:\n
• **Dataset Overview:** The final traffic-weather dataset using selected features\n
• **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
• **Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions""")

with st.expander('View Final Traffic-Weather Dataset'):
      dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Selected_Traffic_Data.csv')
      dataset_df

# User-defined features
with st.sidebar:
      st.header("Input features below to generate a prediction:")
      '''features:
      temperature_2m (°C)
      soil_temperature_0_to_7cm (°C)
      Driving Direction
      apparent_temperature (°C)
      soil_temperature_7_to_28cm (°C)
      surface_pressure (hPa)
      vapour_pressure_deficit (kPa)
      Minute
      DayOfYear'''

      date = st.datetime_input(
      "Choose date and time:",
      datetime.datetime(2025, 11, 19, 16, 45),
      )
      #get hour, minute, day of year

      temp = st.slider("Choose temperature (°C):",0, 27, 17.5)
      #soil_temp_0 = st.slider("Choose soil temperature (0-7 cm):", )
      #driving_direction = st.slider("Choose driving direction: 0 (Backward), 1 (Forward)", 0, 1, 0)
      #app_temp = st.slider("Choose apparent temperature:",)
      #soil_temp_7 = st.slider("Choose soil temperature (7-28 cm):")
      #s_pressure = st.slider("Choose surface pressure:")
      #v_pressure = st.slider("Choose vapour pressure:")
      
      
#CV SPlit
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
X = train_val.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y = train_val['Simulated Traffic Level']
X_test = test.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y_test = test['Simulated Traffic Level']

