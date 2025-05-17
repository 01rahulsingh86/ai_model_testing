from utils.loader import load_data

def test_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Null values found"

def test_valid_income_range():
    df = load_data()
    assert (df['income'] >= 0).all(), "Negative incomes detected"

