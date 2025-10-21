import streamlit as st
import pandas as pd
import pickle

st.title('🏠 House Price Prediction')

st.write('Enter house details:')

sqft = st.number_input('Square footage', 500, 10000, 2000)
bed = st.number_input('Bedrooms', 1, 10, 3)
bath = st.number_input('Bathrooms', 1, 5, 2)

if st.button('Predict'):
    st.success('Prediction feature not implemented yet — placeholder!')

