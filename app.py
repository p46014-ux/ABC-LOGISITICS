import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
# Make sure 'logistic_regression_model_and_scaler.joblib' is in the same directory as app.py
try:
    model_components = joblib.load('logistic_regression_model_and_scaler.joblib')
    model = model_components['model']
    scaler = model_components['scaler']
except FileNotFoundError:
    st.error("Error: 'logistic_regression_model_and_scaler.joblib' not found. Please ensure the file is in the same directory as app.py")
    st.stop()
except KeyError as e:
    st.error(f"Error loading model components: {e}. Make sure the joblib file contains 'model' and 'scaler'.")
    st.stop()

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Define the input features based on the original DataFrame columns
# x.columns: Index(['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
#        'Delivery_Slot', 'Vehicle_Type', 'Driver_Experience', 'Customer_Rating',
#        'Preparation_Time', 'Peak_Hour', 'Delivery_Area', 'Number_of_Items'],
#       dtype='object')

delivery_distance = st.slider('Delivery Distance (km)', 1.0, 100.0, 20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1:Clear, 2:Rainy, 3:Foggy, 4:Snowy, 5:Stormy)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1:Morning, 2:Afternoon, 3:Evening)', 1, 3, 2)
vehicle_type = st.slider('Vehicle Type (1:Bike, 2:Car, 3:Van)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 30, 5)
customer_rating = st.slider('Customer Rating (1-5)', 1.0, 5.0, 4.0)
preparation_time = st.slider('Preparation Time (minutes)', 5, 60, 20)
peak_hour = st.radio('Is it a Peak Hour?', [0, 1], index=0, format_func=lambda x: 'Yes' if x==1 else 'No') # 0 for No, 1 for Yes
delivery_area = st.slider('Delivery Area (1-100, larger value for complex area)', 1, 100, 50)
number_of_items = st.slider('Number of Items', 1, 50, 5)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[delivery_distance,
                            traffic_congestion,
                            weather_condition,
                            delivery_slot,
                            vehicle_type,
                            driver_experience,
                            customer_rating,
                            preparation_time,
                            peak_hour,
                            delivery_area,
                            number_of_items]], 
                            columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
                                     'Delivery_Slot', 'Vehicle_Type', 'Driver_Experience', 
                                     'Customer_Rating', 'Preparation_Time', 'Peak_Hour', 
                                     'Delivery_Area', 'Number_of_Items'])

if st.button('Predict Delivery Delay'):
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('There is a HIGH likelihood of Delivery Delay.')
    else:
        st.success('Delivery is likely to be ON TIME.')
        
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
