import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image

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
    if st.button('Visualize'):
        st.markdown('Cleaning data.....please wait for visualization')

        # Load the dataset
        df = pd.read_csv(r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\co2_emissions (1).csv')

        # Check for null values
        missing_data = df.isnull().sum()

        # Identify duplicate rows
        duplicate_rows = df[df.duplicated()]

        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()

        # One-Hot Encoding for 'transmission' and 'fuel_type'
        df_encoded = pd.get_dummies(df_cleaned, columns=['transmission', 'fuel_type'], drop_first=True)

        # Dropping redundant features like 'model'
        df_final = df_encoded.drop(['model'], axis=1)

        st.write("Missing Data:\n", missing_data)
        st.write("Duplicate Rows Removed:", duplicate_rows.shape[0])
        st.write("Dataset Cleaned\n")

        df_final.to_csv(r'C:\Users\SHASHANK K\PycharmProjects\Project Files\CO2 Emission\Store Room\cleaned_data.csv', index=True)

        st.header("Visualizations")
        ########################################################################

        # Histograms for numerical features
        graph1 = df_final.hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.savefig('visualizations/graph1.png')
        img = Image.open('visualizations/graph1.png')
        st.image(img, use_column_width=True)

########################################################################################################################
        numerical_features = ['engine_size', 'cylinders', 'fuel_consumption_city',
                              'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                              'fuel_consumption_comb(mpg)', 'co2_emissions']

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(numerical_features, 1):
            graph2 = plt.subplot(3, 3, i)
            sns.boxplot(y=df_final[feature])
            plt.title(f'Box Plot of {feature}')
        plt.tight_layout()
        plt.savefig('visualizations/graph2.png')
        img2 = Image.open('visualizations/graph2.png')
        st.image(img2, use_column_width=True)

########################################################################################################################
        plt.figure(figsize=(12, 8))
        corr_matrix = df_final.corr()
        graph3 = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix Heatmap')
        plt.savefig('visualizations/graph3.png')
        img3 = Image.open('visualizations/graph3.png')
        st.image(img3, use_column_width=True)

########################################################################################################################
        plt.figure(figsize=(10, 6))
        graph4 = sns.scatterplot(x='engine_size', y='co2_emissions', data=df_final)
        plt.title('Engine Size vs CO₂ Emissions')
        plt.savefig('visualizations/graph4.png')
        img4 = Image.open('visualizations/graph4.png')
        st.image(img4, use_column_width=True)

########################################################################################################################
        plt.figure(figsize=(10, 6))
        categorical_features = ['fuel_type_E', 'fuel_type_N', 'fuel_type_X', 'fuel_type_Z']

        for feature in categorical_features:
            avg_emissions = df_final.groupby(feature)['co2_emissions'].mean().reset_index()
            graph5 = sns.barplot(x=feature, y='co2_emissions', data=avg_emissions)
            plt.title(f'Average CO₂ Emissions by {feature}')
            plt.savefig(f'visualizations/plot{feature}.png')
            img5 = Image.open(f'visualizations/plot{feature}.png')
            st.image(img5, use_column_width=True)

except AttributeError as ae:
    pass