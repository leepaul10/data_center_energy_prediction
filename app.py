

import streamlit as st
import pandas as pd
import joblib

#Page Configuration
st.set_page_config(page_title="Data Center Energy Prediction", layout="wide")

#Resource Loading
@st.cache_resource
def load_resources():
    data_path = r'C:\Users\leepa\Documents\data_center_energy_prediction\data\processed\feateng_dataset.csv'
    model = joblib.load("final_model.pkl")
    df = pd.read_csv(data_path)
    return df, model

try:
    data, model = load_resources()
except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()

#Header
st.title(" Data Center Energy Prediction ")

#Building Selection (Sidebar)
st.sidebar.header("Building Settings")
building_ids = sorted(data["building_id"].unique())
selected_id = st.sidebar.selectbox("Select Building ID", building_ids)
#Fetch building-specific row
building_row = data[data["building_id"] == selected_id].iloc[0]

#Layout Setup
#Col1: Metadata (Blue Block) | Col2: Prediction Form
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🏢 Building Info")
    #Using st.info for the Blue Block
    with st.container():
        st.info(f"""
            **Metadata (Auto-filled):**
            
            * **Primary Use:** {int(building_row['primary_use'])}
            * **Square Feet:** {int(building_row['square_feet']):,}
            * **Year Built:** {int(building_row['year_built'])}
            * **Site ID:** {int(building_row['site_id'])}
            
            ---
            *Selected Building: {selected_id}*
        """)

with col2:
    with st.form("prediction_form"):
        st.subheader("🌡️ Forecast Parameters")
        
#Meter & Time Inputs
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            meter = st.selectbox("Meter Type", sorted(data["meter"].unique()))
            meter_reading = st.number_input("Current Meter Reading", 0.02, 9070.27, 0.02)
            hour = st.slider("Hour", 0, 23, 12)
        with t_col2:
            month = st.slider("Month", 1, 12, 1)
            day = st.slider("Day of Month", 1, 31, 1)
            weekday = st.slider("Weekday (0=Mon)", 0, 6, 0)

#Weather Inputs
        st.markdown("---")
        st.write("**🌩️ Weather Conditions**")
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            air_temp = st.number_input("Air Temp", -25.6, 47.2, 0.0)
            dew_temp = st.number_input("Dew Temp", -29.4, 25.6, 0.0)
        with w_col2:
            cloud_cov = st.number_input("Cloud Coverage", 0, 9, 0)
            precip = st.number_input("Precip Depth", -1.0, 333.0, 0.0)
        with w_col3:
            sea_level = st.number_input("Sea Pressure", 981.6, 1040.2, 1013.0)
            wind_speed = st.number_input("Wind Speed", 0.0, 16.0, 0.0)
            wind_dir = st.number_input("Wind Direction", 0.0, 360.0, 0.0)

#lag Features
        st.markdown("---")
        st.write("**History (Lags)**")
        l_col1, l_col2, l_col3 = st.columns(3)
        lag_1h = l_col1.number_input("Lag 1h", 0.02, 9070.27, 0.02)
        lag_24h = l_col2.number_input("Lag 24h", 0.03, 10565.30, 0.03)
        lag_48h = l_col3.number_input("Lag 48h", 0.11, 11767.0, 0.11)

        submit = st.form_submit_button("Run Prediction", use_container_width=True)

#Output
if submit:
    #Build DataFrame matching the training feature set
    input_df = pd.DataFrame([[
        selected_id, meter, meter_reading, building_row["site_id"],
        building_row["primary_use"], building_row["square_feet"], 
        building_row["year_built"], air_temp, cloud_cov, dew_temp, 
        precip, sea_level, wind_dir, wind_speed, hour, day, 
        weekday, month, lag_1h, lag_24h, lag_48h
    ]], columns=[
        "building_id", "meter", "meter_reading", "site_id", "primary_use",
        "square_feet", "year_built", "air_temperature", "cloud_coverage",
        "dew_temperature", "precip_depth_1_hr", "sea_level_pressure",
        "wind_direction", "wind_speed", "hour", "day", "weekday", "month",
        "lag_1h", "lag_24h", "lag_48h"
    ])

    #Convert to numeric to avoid object types
    input_df = input_df.apply(pd.to_numeric)
    
    prediction = model.predict(input_df)

    st.divider()
    st.subheader("📊 Forecasted Energy Consumption 📊")
    m1, m2, m3 = st.columns(3)
    m1.metric("T + 1 Hour", f"{prediction[0][0]:.2f} kWh")
    m2.metric("T + 24 Hours", f"{prediction[0][1]:.2f} kWh")
    m3.metric("T + 48 Hours", f"{prediction[0][2]:.2f} kWh")