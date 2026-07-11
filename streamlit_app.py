import streamlit as st
import pandas as pd

st.title('🚗 Traffic Congestion Prediction')

st.write('Description!')

# dataset with CV split
dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Traffic_Dataset.csv')
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
# creating x (selected features only) and y variables
selected_features = ['Hour','temperature_2m (°C)',
                    'soil_temperature_0_to_7cm (°C)',
                    'Driving Direction',
                    'apparent_temperature (°C)',
                    'soil_temperature_7_to_28cm (°C)',
                    'surface_pressure (hPa)',
                    'vapour_pressure_deficit (kPa)',
                    'Minute',
                    'DayOfYear']
X = train_val[selected_features]
y = train_val['Simulated Traffic Level']
X_test = test[selected_features]
y_test = test['Simulated Traffic Level']

X
X_test
