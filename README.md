# 🧠 AI Model Testing Framework

This project provides a full-stack framework to **train, validate, explain, and deploy** an ML model, with features including:

- ✅ Data drift detection
- ⚖️ Bias and explainability tests (using `evidently`)
- 🧪 Model validation
- 🌐 REST API deployment using FastAPI
- 📊 HTML-based test reports

---

## 📂 Project Structure

```bash
.
├── app.py                  # FastAPI backend to serve model
├── model/                  # Trained model artifacts (model.pkl)
├── data/                   # Raw and reference datasets
├── tests/                  # Test scripts and reports
│   └── reports/            # HTML reports (data drift, bias, etc.)
├── utils/                  # Helper utilities (preprocessing, metrics)
├── run_all_tests.sh        # Script to run all tests and generate reports
├── requirements.txt        # Minimal Python dependencies
├── requirements-dev.txt    # Full dependencies (incl. testing and analysis)
└── README.md               # You're here!


⚙️ Setup Instructions
Create a virtual environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

Run model testing
bash run_all_tests.sh
Start API server
uvicorn app:app --reload

🔍 Example API Request
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [5.1, 3.5, 1.4, 0.2]}'




## 🔬 Types of Testing Performed

### 1. ✅ **Unit Testing**
Verifies individual components like model prediction logic and utility functions.

- Test file: `tests/test_accuracy.py`
- Example: Ensures `predict()` returns values of correct type and shape.

---

### 2. 🔁 **Data Validation & Drift Testing**
Checks whether incoming data distributions match the training/reference data using `evidently`.

- Test file: `tests/test_data_validation.py`
- Reports:
  - `data_drift_report.html`
  - `missing_data_report.html`
- Tools: `evidently.test_suite`, `pandas_profiling`

---

### 3. 🧪 **Functional Testing**
Validates core application logic (e.g., model outputs based on input features).

- Test file: `tests/test_functional.py`
- Example: Given a known input, does the model produce an expected output?

---

### 4. 🔗 **Integration Testing**
Ensures different parts of the pipeline (data preprocessing + model + API layer) work cohesively.

- Test file: `tests/test_integration.py`
- Covers the end-to-end predict flow with real API requests and mock inputs.

---

### 5. ⚙️ **Performance Testing**
Measures response time, memory usage, and ensures model latency is acceptable.

- Test file: `tests/test_performance.py`
- Can be extended with `time`, `psutil`, or `memory_profiler`.

---

### 6. ⚖️ **Bias & Fairness Testing**
Checks for bias or performance differences across sensitive attributes (e.g., gender, age).

- Generated Report: `bias_fairness_report.html`
- Framework: `evidently`, `TestFairness`, `TestPredictionBias`

---

### 7. 🔍 **Explainability Testing**
Generates SHAP-style or feature importance insights on how the model makes predictions.

- File: `explainability_report.html` (optional)
- Can be extended with SHAP, LIME, or `evidently.explainers`




📈 Test Reports
After running run_all_tests.sh  youll find:

Data Drift → tests/reports/data_drift_report.html
Bias/Explainability → tests/reports/bias_fairness_report.html
Open in browser to explore visual test summaries.

🚀 Deployment (Optional)
You can deploy this backend API using:
Render
Railway
Docker + AWS/GCP/Azure



👨‍💻 Author
Rahul Singh
GitHub: @01rahulsingh86

📄 License
This project is licensed under the MIT License.

Let me know if you also want to include screenshots, model performance metrics, or badges!
