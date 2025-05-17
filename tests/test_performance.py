import time
from utils.loader import load_model, load_data

def test_inference_time():
    model = load_model()
    data = load_data()
    X = data[['age', 'income']][:1000]

    start = time.time()
    _ = model.predict(X)
    elapsed = time.time() - start

    assert elapsed < 2, f"Inference took too long: {elapsed:.2f}s"

