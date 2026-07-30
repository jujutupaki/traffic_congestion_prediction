import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import joblib
import plotly.express as px

#add dark theme? spotify???
dataset_df = pd.read_csv('https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/Traffic_Data_Selected_Features.csv')

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title('🚗 Traffic Congestion Prediction for Harrison Road, Baguio City')

#CV SPlit
dataset_df['10_Minutes_Interval'] = pd.to_datetime(dataset_df['10_Minutes_Interval'])
train_val = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-01-01 00:00:00')) & (dataset_df['10_Minutes_Interval'] <= '2025-11-23 11:50:00')].reset_index(drop=True)
test = dataset_df[(dataset_df['10_Minutes_Interval'] >= pd.Timestamp('2025-11-24 00:00:00'))].reset_index(drop=True)
X = train_val.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y = train_val['Simulated Traffic Level']
X_test = test.drop(columns=['Simulated Traffic Level', '10_Minutes_Interval'])
y_test = test['Simulated Traffic Level']

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

# User-defined features
with st.sidebar:
      st.header("PLEASE INPUT FEATURES")
      date = st.datetime_input(
      "Select date and time:",
      datetime(2025, 11, 19, 16, 40),
      step=timedelta(minutes=10))
      temp = st.number_input("Select temperature (°C)", value=0.0)
      soil_temp_0 = st.number_input("Select 0-7 cm soil temperature (°C)", value=0.0)
      driving_direction = st.selectbox("Select driving direction (Backward: 0, Forward: 1)", [0, 1])
      app_temp = st.number_input("Select apparent temperature (°C)", value=0.0)
      soil_temp_7 = st.number_input("Select 7-28 cm soil temperature (°C)", value=0.0)
      s_pressure = st.number_input("Select surface pressure (hPa)", value=0.0)
      v_pressure = st.number_input("Select vapour pressure (kPa)", value=0.0)
      date = pd.to_datetime(date)
      min = date.minute
      hour = date.hour
      dayofyear = date.dayofyear
      predict_clicked = st.sidebar.button(
      "Start Prediction",
      use_container_width=True
      )

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

pred_dict = {
    0: "Low Traffic",
    1: "Moderate Traffic",
    2: "Heavy Traffic"
}

def display_prediction(prediction):
    # No prediction yet
    if prediction is None:
        bg_color = "#e9ecef"      # Light gray
        text_color = "#6c757d"    # Dark gray
        prediction_text = "No Prediction Yet"

    else:
        # get predicted class
        prediction = int(prediction[0])

        if prediction == 0:
            bg_color = "#d4edda"      # Light green
            text_color = "#155724"    # Dark green
        elif prediction == 1:
            bg_color = "#fff3cd"      # Light yellow
            text_color = "#856404"    # Dark yellow
        elif prediction == 2:
            bg_color = "#f8d7da"      # Light red
            text_color = "#721c24"    # Dark red
        else:
            bg_color = "#e9ecef"
            text_color = "#6c757d"

        prediction_text = pred_dict[prediction]

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:20px 15px;
            border-radius:12px;
            border:2px solid {text_color};
            margin-bottom:10px;
            height:100%;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
            <h3 style="margin:0; color:{text_color}; font-size:20px;">
                🚦 Predicted Traffic Congestion:
            </h3>
            <h1 style="margin-top:10px; color:{text_color}; font-size:40px;">
                ‎ ‎ ‎ {prediction_text}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    return joblib.load(f"models/XGBoost.pkl")

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if predict_clicked:
    model = load_model()
    st.session_state.prediction = model.predict(input_df)

# Prediction and legends
col1, col2 = st.columns(2)

with col1:
    # Prediction box is always displayed
    # Shows "No Prediction Yet" before the user clicks Start Prediction
    display_prediction(st.session_state.prediction)
    st.write("Start by selecting features in the sidebar. Then, click the 'Start Prediction' button to run the model.")
    with st.expander("CLICK TO VIEW CHOSEN FEATURES: "):
        st.write(f"""**Date and Time:** {date}\n
    ○ Corresponding Minute: {min}\n
    ○ Corresponding Hour: {hour}\n
    ○ Corresponding Day of Year: {dayofyear}\n
**Temperature:** {temp} °C)\n
**0-7 cm Soil Temperature:** {soil_temp_0} °C)\n
**Apparent Temperature:** {app_temp} °C)\n
**7-28 Soil Temperature:** {soil_temp_7} °C)\n
**Surface Pressure:** {s_pressure} hPa\n
**Vapour Pressure:** {v_pressure} kPa
    """)
        
with col2:
    st.info("""**Further Interpretation:**

🟢 **Low:**  
Minimal vehicle volume detected  
Wide gaps between vehicles  
Free-flowing movement  

🟡 **Moderate:**  
Increased vehicle volume detected  
Average and steady moving traffic  
Minor speed reductions  

🔴 **Heavy:**  
Peak vehicle volume detected  
Dense clustering of vehicles  
Delayed traffic speed
""")

st.header("Final Dataset Features & Ranking:", divider="gray")

final_features_df = pd.read_csv(
    "https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/models/final_features.csv")

features_fig = px.bar(
    final_features_df,
    x="Mean Boarda Score",
    y="Features",
    orientation="h",
     color="Mean Boarda Score",
    color_continuous_scale="RdYlGn",
    hover_data=["Votes", "Rank Spread"],
    title="Feature Importance Based on Mean Borda Score"
)

features_fig.update_layout(
    xaxis_title="Mean Borda Score",
    yaxis_title="Features",
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(features_fig)

st.header("Model Evaluation using Classification Metrics:", divider="gray")

metrics_df = pd.read_csv("https://raw.githubusercontent.com/jujutupaki/traffic_congestion_prediction/refs/heads/master/models/metrics_df.csv",
             index_col=0)

col1, col2 = st.columns([1,4])

with col1:
    st.info("Select metric/s:")

    accuracy = st.checkbox("Accuracy")
    precision = st.checkbox("Precision")
    recall = st.checkbox("Recall")
    f1_score = st.checkbox("F1 Score")
    
    st.info("Select model/s:")

    rf = st.checkbox("Random Forest")
    xgb = st.checkbox("XGBoost")
    lstm = st.checkbox("LSTM")

with col2: 
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
        metrics_fig = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".3f",
            title="Model Comparison",
        )
    
        metrics_fig.update_layout(
        yaxis=dict(
            range=[0, 0.82],
            tickmode="array",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8]
            )
        )
    
        st.plotly_chart(metrics_fig, use_container_width=True)
    
    else:
        st.warning("Please select at least one model and one metric.")

st.write("")
st.write("")
st.write("")

st.header("Model Evaluation using Statistical Tools:", divider="gray")

col1, col2 = st.columns([1,3.5])

st.markdown("""
    <style>
    [data-testid="stMetricLabel"] {
        font-size: 20px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

with col1:
    st.subheader("Cochran's Q Test")
    st.metric("Statistic", "22.3586", border=True)
    st.metric("p-value", "<0.001", border=True)
    st.metric("Decision", "Significant", border=True)


def significant(p_value):
    st.markdown(
        f"""
        <div style="
            background-color: #d4edda;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #c3e6cb;
        ">
            <div style="color: #155724; font-size: 16px; font-weight: bold;">
                Significant
            </div>
            <div style="color: #155724; font-size: 28px; font-weight: bold;">
                {p_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def insignificant(p_value):
    st.markdown(
        f"""
        <div style="
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #f5c2c7;
        ">
            <div style="color: #721c24; font-size: 16px; font-weight: bold;">
                Not Significant
            </div>
            <div style="color: #721c24; font-size: 28px; font-weight: bold;">
                {p_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Create DataFrames for each pair
rf_xgb_df = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Model Advantage": [91, 144]
})

rf_lstm_df = pd.DataFrame({
    "Model": ["Random Forest", "LSTM"],
    "Model Advantage": [188, 162]
})

xgb_lstm_df = pd.DataFrame({
    "Model": ["XGBoost", "LSTM"],
    "Model Advantage": [182, 103]
})


with col2:
    st.subheader("Pairwise McNemar's with Bonferroni Correction")

    cola, colb = st.columns([1, 3])

    with cola:
        selected_pair = st.radio(
            "Select a pair:",
            [
                "Random Forest & XGBoost",
                "Random Forest & LSTM",
                "XGBoost & LSTM"
            ]
        )

        # Keep the same Boolean logic
        rf_xgb = selected_pair == "Random Forest & XGBoost"
        rf_lstm = selected_pair == "Random Forest & LSTM"
        xgb_lstm = selected_pair == "XGBoost & LSTM"

        # Random Forest vs XGBoost
        if rf_xgb:
            significant("0.0007")

        # Random Forest vs LSTM
        if rf_lstm:
            insignificant("0.1814")

        # XGBoost vs LSTM
        if xgb_lstm:
            significant("<0.0001")

    with colb:

        # Random Forest vs XGBoost
        if rf_xgb:
            fig = px.bar(
                rf_xgb_df,
                x="Model",
                y="Model Advantage",
                text="Model Advantage",
                title="Random Forest vs XGBoost"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Model Advantage",
                xaxis_title="Model",
                yaxis=dict(range=[0, 160])
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # Random Forest vs LSTM
        if rf_lstm:
            fig = px.bar(
                rf_lstm_df,
                x="Model",
                y="Model Advantage",
                text="Model Advantage",
                title="Random Forest vs LSTM"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Model Advantage",
                xaxis_title="Model",
                yaxis=dict(range=[0, 210])
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # XGBoost vs LSTM
        if xgb_lstm:
            fig = px.bar(
                xgb_lstm_df,
                x="Model",
                y="Model Advantage",
                text="Model Advantage",
                title="XGBoost vs LSTM"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Model Advantage",
                xaxis_title="Model",
                yaxis=dict(range=[0, 210])
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
