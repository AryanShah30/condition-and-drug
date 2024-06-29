import pickle
import streamlit as st
import pandas as pd
import os

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "drugsComTrain.csv")
if not os.path.exists(file_path):
    st.error(f"Dataset file '{file_path}' not found.")
    st.stop()

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

# Continue with the rest of your Streamlit application code...
