import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Delivery_Delay.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the features below to predict if a delivery will be delayed.')

# Input features from the user
delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 50.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5, 5 being worst)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (Years)', 0, 30, 10)
num_stops = st.slider('Number of Stops', 0, 20, 5)
vehicle_age = st.slider('Vehicle Age (Years)', 0, 15, 5)
road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 0.0, 30.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a button for prediction
if st.button('Predict Delivery Delay'):
    # Prepare the input array for the model
    # Ensure the order matches the features used during training
    input_features = np.array([
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error('The delivery is predicted to be DELAYED.')
    else:
        st.success('The delivery is predicted to be ON TIME.')

    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
