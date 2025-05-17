from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Sample training data
data = pd.DataFrame({
    'age': [25, 40, 35, 23],
    'income': [50000, 100000, 70000, 30000],
    'loan_approved': [1, 1, 1, 0]
})

X = data[['age', 'income']]
y = data['loan_approved']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model/loan_model.pkl')
print("✅ Model trained and saved to model/loan_model.pkl")
