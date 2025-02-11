import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

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

try:

    if st.button('Start Training'):
        df_final = pd.read_csv(r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\cleaned_data.csv')

        # Normalize numerical features using Min-Max Scaling
        scaler = MinMaxScaler()
        numerical_features = ['engine_size', 'cylinders', 'fuel_consumption_city',
                              'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                              'fuel_consumption_comb(mpg)', 'co2_emissions']

        df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])

        # Create new feature: Fuel Efficiency Ratio
        # (Combining fuel consumption in city and highway for a more balanced metric)
        df_final['fuel_efficiency_ratio'] = df_final['fuel_consumption_city'] / (df_final['fuel_consumption_hwy'] + 1e-6)

        # Define features (X) and target (y)
        X = df_final[['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)', 'fuel_consumption_comb(mpg)']]
        y = df_final['co2_emissions']

        # Split the dataset into Training (80%) and Testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #############################################################
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        joblib.dump(lr, r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\Linear_Regression.h5')

        #############################################################
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        joblib.dump(dt, r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\Decision_Tree.h5')

        ##############################################################
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        joblib.dump(rf, r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\Random_Forest.h5')

        ###############################################################
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        joblib.dump(gb, r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\Gradient_Boost.h5')

        ###############################################################
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        joblib.dump(xgb, r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\XGBoost_Regressor.h5')

        ################################################################

        def evaluate_model(name, y_test, y_pred):
            st.header(f"{name} Performance:")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
            st.write(f"R2 Score: {r2_score(y_test, y_pred)}\n")

        st.header("Training Completed: Model Evaluation")
        evaluate_model("Linear Regression", y_test, y_pred_lr)
        evaluate_model("Decision Tree Regressor", y_test, y_pred_dt)
        evaluate_model("Random Forest Regressor", y_test, y_pred_rf)
        evaluate_model("Gradient Descent", y_test, y_pred_gb)
        evaluate_model("XGBoost Regressor", y_test, y_pred_xgb)

###################################################################################################################
        model_scores = {
            "Linear Regression": r2_score(y_test, y_pred_lr),
            "Decision Tree Regressor": r2_score(y_test, y_pred_dt),
            "Random Forest Regressor": r2_score(y_test, y_pred_rf),
            "Gradient_Boost": r2_score(y_test, y_pred_gb),
            "XGBoost_Regressor": r2_score(y_test, y_pred_xgb)
        }
        best_model = max(model_scores, key=model_scores.get)

        st.info(f"\nBest performing model: **{best_model}** with **{(model_scores[best_model])*100}%** Accuracy")

except AttributeError as ae:
    pass
