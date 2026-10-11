import streamlit as st
import numpy as np
import pickle

with open("iris_dataset.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris flower prediction")
sepal_length = st.slider("Sepal Length (in cm)", 4.0, 8.0)
sepal_width = st.slider("Sepal Width (in cm)", 2.0, 5.0)
petal_length = st.slider("Petal Length (in cm)", 1.0, 7.0)
petal_width = st.slider("Petal Width (in cm)", 1.0, 3.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width ]])
    prediction = model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Iris species: {species[prediction[0]]}")