import joblib
import pandas as pd

def load_model():
    return joblib.load("model/loan_model.pkl")

def load_data():
    return pd.read_csv("data/test_data.csv")

