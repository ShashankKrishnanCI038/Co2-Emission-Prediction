import streamlit as st

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

st.info("""The CO₂ Emission Prediction using **Machine Learning** project aims to estimate carbon dioxide emissions based on vehicle attributes such as engine size, fuel consumption, and other factors. 
By leveraging machine learning algorithms like **Linear Regression**, **Decision Trees**, **Random Forest**, **Gradient Boosting**, and **XGBoost**, the model provides accurate predictions to help analyze and reduce emissions. 
This project can assist in environmental impact assessments and support sustainable decision-making in the automotive industry.""")

st.markdown("""Machine Learning model used in this project are:\n
1. Linear Regression
2. Decision Tree
3. Random Forest
4. Gradient Descent
5. XGBoost Regressor""")

st.markdown(""" In this project, we use multiple machine learning models to predict CO₂ emissions based on vehicle characteristics. 
Each model has its strengths, allowing us to compare their performances and choose the most accurate one. 

**1. Linear Regression**\n
**Why Used?**
Linear Regression is a simple yet effective model that establishes a linear relationship between independent variables (features like engine size, fuel consumption) and the dependent variable (CO₂ emissions).

**How It Works?**
It fits a straight line (best-fit line) through the data points using the equation: **y=mx+c**
Minimizes the difference between actual and predicted values using the least squares method.
_____________________________________________________________________________________________________
**2. Decision Tree Regressor**\n
**Why Used?**
Decision Trees split the dataset into smaller subsets based on conditions, making them effective for capturing non-linear relationships in CO₂ emissions data.

**How It Works?**\n
Divides data into different branches based on feature values\n
Reaches a final prediction at the leaves of the tree\n
Uses if-else conditions to make decisions
_____________________________________________________________________________________________________
**3. Random Forest Regressor**\n
**Why Used?**
Random Forest is an ensemble model that combines multiple Decision Trees to improve accuracy and reduce overfitting.

**How It Works?**\n
Creates multiple Decision Trees using different subsets of the dataset
Averages the predictions from all trees to get a final result
Uses Bootstrap Aggregation (Bagging) to reduce variance
_____________________________________________________________________________________________________
**4. Gradient Boosting Regressor**
**Why Used?**
Gradient Boosting is another ensemble learning technique, but unlike Random Forest, it builds trees sequentially, with each tree correcting the errors of the previous one.

**How It Works?**\n
Starts with a weak model (usually a Decision Tree)
Builds new trees that focus on the errors of previous trees
Uses Gradient Descent to minimize errors
_____________________________________________________________________________________________________
**5. XGBoost Regressor**\n
**Why Used?**
XGBoost (Extreme Gradient Boosting) is an optimized version of Gradient Boosting that is faster and more efficient. It is widely used in machine learning competitions due to its performance.

**How It Works?**\n
Uses advanced techniques like regularization and parallel processing
Corrects errors more efficiently compared to traditional Gradient Boosting
Handles missing values and large datasets well
""")