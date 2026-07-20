import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import joblib
import plotly.express as px

st.markdown("""
<style>
[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 0rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
st.set_page_config(
    layout="wide",
    page_title="Traffic Congestion Prediction",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

st.title('🚗 Traffic Congestion Prediction')

st.info("""Welcome to **brr-traffic.streamlit.app**, the interactive dashboard for our thesis: \n
"Predicting Traffic Congestion under Different Weather Conditions Using Machine Learning Approaches"

What you can explore here:\n
🔴 **Dataset Overview:** The final traffic-weather dataset using selected features\n
🟡 **Model Evaluation Dashboard:** Compare performance metrics across the Random Forest, XGBoost, and LSTM models\n
🟢 **Traffic Predictor:** Input custom weather conditions, dates, and times to generate real-time congestion predictions\n""")

#dataset
dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Traffic_Data_Selected_Features.csv')
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

#sidebar size
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 450px
    </style>
    """,
    unsafe_allow_html=True,
)

# User-defined features
with st.sidebar:
      st.header("PLEASE INPUT FEATURES")
      date = st.datetime_input(
      "Select date and time:",
      datetime(2025, 11, 19, 16, 40),
      step=timedelta(minutes=10))
      temp = st.number_input("Select temperature (°C)", value=0.0)
      soil_temp_0 = st.number_input("Select soil temperature (0-7 cm)", value=0.0)
      driving_direction = st.selectbox("Select driving direction (Backward: 0, Forward: 1)", [0, 1])
      app_temp = st.number_input("Select apparent temperature (°C)", value=0.0)
      soil_temp_7 = st.number_input("Select soil temperature (7-28 cm)", value=0.0)
      s_pressure = st.number_input("Select surface pressure (hPa)", value=0.0)
      v_pressure = st.number_input("Select vapour pressure (kPa)", value=0.0)
      date = pd.to_datetime(date)
      min = date.minute
      hour = date.hour
      dayofyear = date.dayofyear

#df for input features
df_label = {
    'Hour': hour,
    'soil_temperature_0_to_7cm (°C)': soil_temp_0,
    'temperature_2m (°C)': temp,
    'Driving Direction': driving_direction,
    'apparent_temperature (°C)': app_temp,
    'soil_temperature_7_to_28cm (°C)': soil_temp_7,
    'surface_pressure (hPa)': s_pressure,
    'vapour_pressure_deficit (kPa)': v_pressure,
    'DayOfYear': dayofyear,
    'Minute': min
}

input_df = pd.DataFrame(df_label, index=[0])
st.info("""Click the button on the top-left corner to expand the sidebar and generate a prediction!\n
Current input for features:""")
input_df

def display_prediction(prediction):
    # map predictions
    pred_dict = {
        0: "Low Traffic",
        1: "Moderate Traffic",
        2: "Heavy Traffic"
    }

    # get predicted class
    prediction = int(prediction[0])

    # color settings
    if prediction == 0:
        bg_color = "#d4edda"      # Light green
        text_color = "#155724"    # Dark green
    elif prediction == 1:
        bg_color = "#fff3cd"      # Light yellow
        text_color = "#856404"    # Dark yellow
    else:
        bg_color = "#f8d7da"      # Light red
        text_color = "#721c24"    # Dark red

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:25px;
            border-radius:12px;
            text-align:center;
            border:2px solid {text_color};
             margin-bottom:30px;
        ">
            <h3 style="margin:0; color:{text_color};">
                🚦Predicted Traffic Congestion:
            </h3>
            <h1 style="margin-top:10px; color:{text_color};">
                {pred_dict[prediction]}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    return joblib.load(f"models/XGBoost.pkl")

#button style
st.markdown("""
<style>
div[data-testid="stButton"] > button {
    background-color: white !important;
    color: black !important;
    border: 2px solid #d0d0d0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

div[data-testid="stButton"] > button:hover {
    background-color: #f5f5f5 !important;
    border-color: #999999 !important;
    color: black !important;
}

div[data-testid="stButton"] > button:focus {
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Start Prediction", use_container_width=True):
    model = load_model()
    st.session_state.prediction = model.predict(input_df)

if st.session_state.prediction is not None:
    display_prediction(st.session_state.prediction)

metrics_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/models/metrics_df.csv',
             index_col=0)

st.info("Select metrics to display:")

accuracy = st.checkbox("Accuracy")
precision = st.checkbox("Precision")
recall = st.checkbox("Recall")
f1_score = st.checkbox("F1 Score")

st.info("Select model/s to show its performance:")

rf = st.checkbox("Random Forest")
xgb = st.checkbox("XGBoost")
lstm = st.checkbox("LSTM")

# Collect active selections
selected_models = []
selected_metrics = []
if rf: selected_models.append("Random Forest")
if xgb: selected_models.append("XGBoost")
if lstm: selected_models.append("LSTM")
if accuracy: selected_metrics.append("Accuracy")
if precision: selected_metrics.append("Precision")
if recall: selected_metrics.append("Recall")
if f1_score: selected_metrics.append("F1 Score")

if selected_models and selected_metrics:

    # Filter dataframe
    filtered_df = metrics_df.loc[selected_models, selected_metrics]

    # Convert to long format for Plotly
    plot_df = filtered_df.reset_index()

    id_col = plot_df.columns[0]

    plot_df = plot_df.melt(
    id_vars=id_col,
    var_name="Metric",
    value_name="Score"
    ).rename(columns={id_col: "Model"})

    # Create grouped bar chart
    fig = px.bar(
        plot_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".3f",
        title="Model Performance Comparison"
    )

    fig.update_layout(yaxis_range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please select at least one model and one metric.")
