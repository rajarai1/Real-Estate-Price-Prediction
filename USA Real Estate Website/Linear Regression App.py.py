import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data 
def load_data():
    return pd.read_csv('USA_Housing.csv')

# Train the model
@st.cache_data 
def train_model(df):
    X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict function
def predict_price(model, inputs):
    return model.predict([inputs])[0]

def main():
    st.title('House Price Prediction App')
    st.write('Enter the details below to predict the house price.')

    # Input fields for the features
    avg_income = st.number_input('Average Area Income', min_value=0.0, value=50000.0)
    house_age = st.number_input('Average Area House Age', min_value=0.0, value=5.0)
    num_rooms = st.number_input('Average Area Number of Rooms', min_value=0.0, value=6.0)
    num_bedrooms = st.number_input('Average Area Number of Bedrooms', min_value=0.0, value=3.0)
    population = st.number_input('Area Population', min_value=0.0, value=30000.0)

    # Load data and train model
    df = load_data()
    model = train_model(df)

    # Make prediction
    if st.button('Predict'):
        inputs = [avg_income, house_age, num_rooms, num_bedrooms, population]
        predicted_price = predict_price(model, inputs)
        st.write(f'The predicted house price is: ${predicted_price:,.2f}')

if __name__ == '__main__':
    main()
