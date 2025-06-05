import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered", initial_sidebar_state="auto")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 1, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Preprocessing input
sex = 1 if sex == 'male' else 0
embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("❌ Did not survive.")
