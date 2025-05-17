from utils.loader import load_model
import pandas as pd


def test_predict_valid_input():
    model = load_model()
    input_data = pd.DataFrame([[30, 60000]], columns=['age', 'income'])
    prediction = model.predict(input_data)[0]
    assert prediction in [0, 1]
