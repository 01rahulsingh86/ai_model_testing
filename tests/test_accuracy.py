from utils.loader import load_model, load_data
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = load_model()
    data = load_data()
    # Add ground truth labels to test_data.csv beforehand
    X = data[['age', 'income']]
    y_true = data['loan_approved']
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    assert acc > 0.7, f"Accuracy too low: {acc}"

