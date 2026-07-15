import streamlit as st
import pandas as pd
import datetime
import joblib

dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Traffic_Data_Selected_Features.csv')

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚗",
    initial_sidebar_state="collapsed",
)

st.title('🚗 Traffic Congestion Prediction')

st.info("""Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: \n
"Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches".

What you can explore here:\n
🔴 **Dataset Overview:** The final traffic-weather dataset using selected features\n
🟡 **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
🟢 **Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions\n""")

#dataset
with st.expander('View Final Traffic-Weather Dataset'):
    st.write(dataset_df.head(10))

#CV SPlit
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
X = train_val.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y = train_val['Simulated Traffic Level']
X_test = test.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y_test = test['Simulated Traffic Level']

# User-defined features
with st.sidebar:
      st.header("Input features:")
      date = st.datetime_input(
      "1. Select date and time:",
      datetime.datetime(2025, 11, 19, 16, 45),
      )
      date = pd.to_datetime(date)
      min = date.minute
      hour = date.hour
      dayofyear = date.dayofyear
      
      temp = st.slider("2. Select temperature (°C):", 0.0, 30.0, 17.5)
      soil_temp_0 = st.slider("3. Select soil temperature (0-7 cm):", 0.0, 30.0, 18.6)
      driving_direction = st.selectbox("4. Driving Direction (Backward: 0, Forward: 1)", [0, 1])
      app_temp = st.slider("5. Select apparent temperature (°C):", 0.0, 27.0, 18.8)
      soil_temp_7 = st.slider("6. Select soil temperature (7-28 cm):", 0.0, 30.0, 18.9)
      s_pressure = st.slider("7. Select surface pressure:", 800.0, 860.0, 846.5)
      v_pressure = st.slider("8. Select vapour pressure:", 0.0, 2.0, 1.88)

#df for input features
df_label = {
    'Hour': hour,
    'Minute': min,
    'DayofYear': dayofyear,
    'temperature_2m (°C)': temp,
    'soil_temperature_0_to_7cm (°C)': soil_temp_0,
    'Driving Direction': driving_direction,
    'apparent_temperature (°C)': app_temp,
    'soil_temperature_7_to_28cm (°C)': soil_temp_7,
    'surface_pressure (hPa)': s_pressure,
    'vapour_pressure_deficit (kPa)': v_pressure
}

input_df = pd.DataFrame(df_label, index=[0])
st.info("""Click the button on the top-left corner to expand the sidebar and generate a prediction!\n
Current input for features:""")
input_df

@st.cache_resource
def load_xgb_model():
    return joblib.load(models/"XGBoost.pkl")

xgb_model = load_xgb_model()

xgb_prediction = xgb_model.predict(input_df)
st.write(xgb_prediction)
