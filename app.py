from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd


app = FastAPI()

# ✅ Allow requests from all origins (or restrict to specific domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://localhost:5500"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/loan_model.pkl")  # Update path if needed

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data['input']], columns=["age", "income"])  # adjust if more features
    result = model.predict(input_df)
    return {"prediction": result.tolist()}

