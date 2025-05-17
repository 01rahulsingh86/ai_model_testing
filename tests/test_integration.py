from utils.loader import load_model, load_data

def test_pipeline_output_shape():
    model = load_model()
    df = load_data()
    X = df[['age', 'income']]
    preds = model.predict(X)
    assert len(preds) == len(X)

