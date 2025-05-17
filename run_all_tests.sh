#!/bin/bash

set -e

echo "🔁 Running all AI model diagnostic tests..."
echo "=========================================="

# Ensure reports directory exists
mkdir -p tests/reports

# -----------------------------
# 📊 Data Drift Detection
# -----------------------------
echo "📊 Running data drift test..."
python3 -c "
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
import pandas as pd

df = pd.read_csv('data/test_data.csv')
ref = df.sample(frac=0.5, random_state=1)
cur = df.sample(frac=0.5, random_state=2)

suite = TestSuite(tests=[DataDriftTestPreset()])
suite.run(reference_data=ref, current_data=cur)
suite.save_html('tests/reports/data_drift_report.html')
print('✅ Drift report saved to tests/reports/data_drift_report.html')
"

# -----------------------------
# ⚖️ Bias / Fairness Check
# -----------------------------
echo "⚖️  Running fairness/bias test..."
python3 -c "
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from utils.loader import load_model, load_data
import numpy as np

model = load_model()
data = load_data()

# Simulate gender for bias check
np.random.seed(42)
data['gender'] = np.random.choice(['male', 'female'], size=len(data))

y_true = data['loan_approved']
y_pred = model.predict(data[['age', 'income']])

frame = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=data['gender'])
print('📊 Selection rate by gender:')
print(frame.by_group)

parity_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=data['gender'])
print(f'📉 Demographic parity difference: {parity_diff}')
"

# -----------------------------
# 🧠 SHAP Explainability
# -----------------------------
echo "🧠 Generating SHAP explainability plot..."
python3 -c "
import shap
from utils.loader import load_model, load_data
import matplotlib.pyplot as plt

model = load_model()
data = load_data()[['age', 'income']]

explainer = shap.Explainer(model.predict, data)
shap_values = explainer(data)

shap.plots.beeswarm(shap_values, max_display=5, show=False)
plt.title('SHAP Beeswarm (Top 5)')
plt.savefig('tests/reports/shap_beeswarm.png')
print('✅ SHAP beeswarm saved to tests/reports/shap_beeswarm.png')
"

echo ""
echo "🎉 All tests completed successfully!"
echo "📁 Reports:"
echo "  - Drift report:      tests/reports/data_drift_report.html"
echo "  - SHAP explanation:  tests/reports/shap_beeswarm.png"

