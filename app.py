import pickle
import streamlit as st
import pandas as pd
import os

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "drugsComTrain.csv")
df = pd.read_csv(file_path)

# Load the prediction model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "patient_classification.sav")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer_3.sav")

try:
    with open(model_path, "rb") as f_model, open(vectorizer_path, "rb") as f_vectorizer:
        prediction_model = pickle.load(f_model)
        prediction_vec_model = pickle.load(f_vectorizer)
except FileNotFoundError:
    st.error(f"Model or vectorizer file not found at '{model_path}' or '{vectorizer_path}'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit application
st.set_page_config(page_title="Condition & Drug", page_icon="💊", layout="wide")
st.title("Patient Condition Classification and Drug Recommendation")
st.markdown("""
    <div style='color: gray; font-size: 15px;'>
        Last updated on 29/06/2024.
    </div>
    """, unsafe_allow_html=True)
st.write("""
This tool helps you by predicting your medical condition based on a description of your symptoms. 
Currently, it supports the following conditions:
- Birth Control
- Depression
- High Blood Pressure
- Diabetes Type 2

Based on the predicted condition, the app will recommend the top three drugs with the highest ratings and usefulness. 
""")

# Text input and button
st.subheader("Enter your condition description below:")
inp = st.text_area(label="",placeholder="Describe your symptoms or condition here...", height=150)
btn = st.button("Check")

if btn:
    if inp.strip():  # Check if input is not empty or whitespace
        # Perform prediction
        transformed_input = prediction_vec_model.transform([inp])
        prediction = prediction_model.predict(transformed_input)[0]  # Get the predicted condition

        # Display predicted condition as a sub header with only the condition name in red using Markdown
        st.markdown(f"### Predicted Condition: <span style='color:red;'>{prediction}</span>", unsafe_allow_html=True)

        # Define function to extract top drugs for a given condition
        def top_drugs_extraction(condition):
            df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100) & (df['condition'] == condition)]
            top_drugs = df_top.sort_values(by=['rating', 'usefulCount'], ascending=[False, False])['drugName'].head(3).tolist()
            return top_drugs

        # Display top recommended drugs if condition is one of the specified ones
        if prediction in ['Birth Control', 'Depression', 'High Blood Pressure', 'Diabetes Type 2']:
            top_drugs = top_drugs_extraction(prediction)
            st.subheader("Top 3 Recommended Drugs:")
            for drug in top_drugs:
                st.write(f"- {drug}")
        else:
            st.subheader("Top 3 Recommended Drugs:")
            st.write("No recommendations available for this condition.")
    else:
        st.warning("Please enter a valid condition description.")
