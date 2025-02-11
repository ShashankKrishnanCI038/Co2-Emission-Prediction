import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="CO2 Emission Prediction",
    page_icon=None
)
# Set the background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image URL or local path
set_background("https://www.pixground.com/wp-content/uploads/2023/12/Ethereal-White-Butterfly-AI-Generated-4K-Wallpaper.jpg")

st.header("CO2 Emission Prediction")
st.markdown("Please Enter the following data for prediction")

try:
    engine_size = st.number_input("Enter the size of engine", placeholder="0 - 10", min_value=0, max_value=10)
    cylinders = st.number_input("Enter the cylinders in engine", placeholder="3 - 16", min_value=3, max_value=16)
    fuel_consumption_city = st.number_input("Enter City Fuel Consumption", placeholder="0 - 100", min_value=0, max_value=100)
    fuel_consumption_hwy = st.number_input("Enter Highway Fuel Consumption", placeholder="0 - 100", min_value=0, max_value=100)
    fuel_consumption_comb1 = st.number_input("Enter Combined Fuel Consumption (L/100 Km)", placeholder="0 - 100", min_value=0, max_value=100)
    fuel_consumption_comb2 = st.number_input("Enter Combined Fuel Consumption (MPG)", placeholder="0 - 100", min_value=0, max_value=100)

    random_values = np.array([engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb1, fuel_consumption_comb2])  # Adjust based on feature count
    random_values = random_values.reshape(1, -1)

    model = joblib.load(r"C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\Random_Forest.h5")
    prediction = model.predict(random_values)

    if st.button('predict'):
        st.write("You entered: ", random_values)
        st.subheader(f"**{prediction[0] * 100}%** of Co2 is emitted from respective engine")

except AttributeError as ae:
    pass

except FileNotFoundError as fe:
    st.write("Trained Model not loaded")