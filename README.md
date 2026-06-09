
#Machine Learning Project


# 🎓 Student Dropout Prediction

A machine learning pipeline that predicts whether a student is at risk of dropping out, using academic performance, engagement metrics, and socioeconomic factors. Built with scikit-learn, imbalanced-learn, and a clean OOP design.

---

## 📌 Overview

Student dropout is a costly problem for institutions and learners alike. This project trains and compares three classifiers — Logistic Regression, Random Forest, and Gradient Boosting — with SMOTE-based class balancing and hyperparameter tuning via `GridSearchCV`. The best-performing model is saved for use on new student records.

---

## 🗂️ Project Structure

```
student_dropout_prediction/
│
├── student_dropout_prediction.ipynb   # Main notebook
├── dataset.csv                        # Input dataset (not included)
├── dropout_predictor.pkl              # Saved model (generated on run)
└── dropout_prediction_results.png    # Visualizations (generated on run)
```

---

## ⚙️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Load Data | Reads CSV, prints shape, target distribution, and missing value counts |
| 2. Preprocess | Encodes target (Dropout=1, Graduate/Enrolled=0), label-encodes categoricals, clips outliers via IQR |
| 3. Feature Engineering | Creates performance ratios, engagement ratios, age groups, economic stability index |
| 4. Feature Selection | Selects top 20 features using Random Forest importances |
| 5. Train Models | Trains 3 classifiers inside SMOTE pipelines with 5-fold CV grid search |
| 6. Evaluate | Compares models by CV F1, Test F1, and AUC; selects best |
| 7. Visualize | Plots feature importances, confusion matrix, ROC curve, and predicted distribution |
| 8. Save | Exports model + encoders to `.pkl` via `joblib` |

---

## 🔧 Engineered Features

- `sem1_performance_ratio` / `sem2_performance_ratio` — approved / (enrolled + 1) per semester
- `overall_performance` — average of semester performance ratios
- `sem1_engagement` / `sem2_engagement` — evaluations / (enrolled + 1)
- `age_group` — binned age at enrollment (≤20, 21–25, 26–30, 30+)
- `economic_stability` — composite of tuition up-to-date, non-debtor, scholarship status

---

## 📊 Models & Evaluation

| Model | Tuned Hyperparameters |
|---|---|
| Logistic Regression | `C`, `max_iter` |
| Random Forest | `n_estimators`, `max_depth` |
| Gradient Boosting | `n_estimators`, `learning_rate` |

All models are wrapped in an `imblearn.Pipeline` with `StandardScaler` → `SMOTE` → Classifier. Best model is selected by **Test F1 score**.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn joblib
```

### Run

1. Place your `dataset.csv` in the working directory (or update the path in `load_data()`).
2. Open and run `student_dropout_prediction.ipynb` end-to-end.
3. Outputs: `dropout_predictor.pkl` and `dropout_prediction_results.png`.

### Dataset Format

The notebook expects a CSV with a `Target` column containing `"Dropout"`, `"Graduate"`, or `"Enrolled"` labels, along with academic and demographic features including:

- `Curricular units 1st/2nd sem (enrolled, approved, evaluations, grade)`
- `Previous qualification (grade)`, `Admission grade`
- `Age at enrollment`
- `Tuition fees up to date`, `Debtor`, `Scholarship holder`

> A compatible dataset is available on [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

---

## 🔮 Predicting New Students

After training, use the `predict_new_student()` method with a dictionary of student features:

```python
import joblib

artifacts = joblib.load('dropout_predictor.pkl')

new_student = {
    'Age at enrollment': 22,
    'Tuition fees up to date': 1,
    'Debtor': 0,
    'Scholarship holder': 1,
    # ... other features
}

predictor.predict_new_student(new_student)
# 🎯 Prediction: Low Risk
# Dropout Probability: 18.42%
```

---

## 📈 Output Visualizations

The notebook generates a 2×2 figure saved as `dropout_prediction_results.png`:

- **Top 10 Feature Importances** (bar chart)
- **Confusion Matrix** (heatmap)
- **ROC Curve** with AUC score
- **Predicted Dropout Distribution** (pie chart)

---

## 🛠️ Tech Stack

- **Python 3.x**
- `scikit-learn` — models, preprocessing, evaluation
- `imbalanced-learn` — SMOTE, pipeline
- `pandas` / `numpy` — data handling
- `matplotlib` / `seaborn` — visualization
- `joblib` — model persistence

---

## 📄 License

This project is released under the MIT License.
