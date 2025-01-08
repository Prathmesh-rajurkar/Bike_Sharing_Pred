import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.bike_share_pred.components.data_ingestion import DataIngestion
from src.bike_share_pred.components.data_transformation import DataTransformation
from src.bike_share_pred.components.model_trainer import ModelTrainer
# from src.bike_share_pred.components.model_evaluation import ModelEval
from src.bike_share_pred.utils.helper_functions import load_model

def main():
    st.set_page_config(page_title="Bike Sharing Prediction", layout="wide")
    
    # Title and Description
    st.title("Bike Sharing Demand Prediction")
    st.write("This app predicts the number of bikes that will be rented based on various features.")
    
    
    st.header("Make Predictions")
    
    # Input form for predictions
    st.subheader("Enter Features for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        season = st.selectbox("Season", [1, 2, 3, 4])
        holiday = st.selectbox("Holiday", [0, 1])
        workingday = st.selectbox("Working Day", [0, 1])
        weather = st.selectbox("Weather", [1, 2, 3, 4])
        
    with col2:
        temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
        humidity = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
        windspeed = st.slider("Wind Speed (Normalized)", 0.0, 1.0, 0.5)
        registered = st.slider("Registered Users", 0, 1000, 500)
        casual = st.slider("Casual Users", 0, 1000, 500)
    
    # Make prediction
    if st.button("Predict"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'season': [season],
            'holiday': [holiday],
            'workingday': [workingday],
            'weathersit': [weather],
            'temp': [temp],
            'hum': [humidity],
            'windspeed': [windspeed],
            'registered': [registered],
            'casual': [casual]
        })
        
        # Load model and make prediction
        model = load_model('artifacts\models\model.pkl')
        result = model.predict(input_data)
        
        st.success(f"Predicted number of bike rentals: {int(result[0])}")

if __name__ == "__main__":
    main()