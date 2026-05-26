# Pune House Price Predictor : 
https://house-price-predictor-t.streamlit.app

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://house-price-predictor-t.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/Built%20with-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)

**🔗 Try it live:** https://house-price-predictor-t.streamlit.app

Linear regression for Pune housing prices, **implemented from scratch in NumPy** to demonstrate the fundamentals from Andrew Ng's ML Specialization (Course 1): cost function, batch gradient descent, vectorization, and z-score feature scaling. A scikit-learn baseline is included only to verify the from-scratch implementation is correct.

A Streamlit UI takes user preferences (locality, BHK, bathrooms, sqft) and returns:
1. A predicted price from the trained model.
2. Matching listings from the dataset, sorted by `listed − predicted` so undervalued listings surface first.

## What I built from scratch

| Concept                | Where                                                     |
| ---------------------- | --------------------------------------------------------- |
| Cost function (MSE)    | `compute_cost` in [src/linear_regression.py](src/linear_regression.py) |
| Gradient of J(w, b)    | `compute_gradients` in [src/linear_regression.py](src/linear_regression.py) |
| Vectorized prediction  | `X @ w + b` (no Python loops over examples)               |
| Batch gradient descent | `LinearRegressionGD.fit` in [src/linear_regression.py](src/linear_regression.py) |
| Z-score feature scaling | [src/scaler.py](src/scaler.py)                            |
| Train/test split       | `train_test_split` in [src/train.py](src/train.py) (NumPy permutation) |
| Sigmoid + BCE cost + L2 (Week 3) | `sigmoid`, `compute_cost`, `compute_gradients` in [src/logistic_regression.py](src/logistic_regression.py) |
| Logistic regression GD | `LogisticRegressionGD` in [src/logistic_regression.py](src/logistic_regression.py) |

## Project layout

```
src/
  linear_regression.py   # from-scratch model
  scaler.py              # z-score scaler
  preprocess.py          # cleaning + one-hot encoding
  train.py               # training entry point
  evaluate.py            # sklearn baseline comparison
  diagnostics.py         # the four diagnostic plots
app.py                   # Streamlit UI
data/raw/                # drop the Kaggle CSV here
models/                  # saved weights, scaler, feature columns
notebooks/figures/       # diagnostic plots (for the LinkedIn post)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

This project uses the **Pune House Data** dataset from Kaggle (works with the well-known "Bengaluru House Data" too — same schema, identical methodology). Download the CSV and save it as `data/raw/pune_houses.csv`.

Expected columns: `area_type, availability, site_location` (or `location`), `size, society, total_sqft, bath, balcony, price`.

## Train

```bash
# Regressor (Course 1 Weeks 1–2)
python -m src.train --csv data/raw/pune_houses.csv

# Classifier (Course 1 Week 3) — must run AFTER the regressor
python -m src.train_classifier
```

The regressor writes `models/weights.npz`, `models/scaler.npz`, `models/feature_columns.json`, and `data/processed/pune_clean.csv`. The classifier writes `models/classifier_weights.npz`, `models/classifier_scaler.npz`, and `models/classifier_meta.json` and labels listings as **Expensive (> ₹100L)** vs **Affordable** with L2-regularized logistic regression.

## Verify against scikit-learn

```bash
python -m src.evaluate --csv data/raw/pune_houses.csv          # regressor
python -m src.evaluate_classifier                              # classifier
```

`src.evaluate` prints MAE / RMSE / R² for both the from-scratch model and `sklearn.linear_model.LinearRegression`. `src.evaluate_classifier` prints accuracy / precision / recall / F1 for the from-scratch classifier and `sklearn.linear_model.LogisticRegression`. Metrics should match closely — the proof both from-scratch implementations are correct.

## Generate diagnostic plots

```bash
python -m src.diagnostics --csv data/raw/pune_houses.csv
```

Produces four plots in `notebooks/figures/`:
1. **Cost vs. iteration** — gradient descent converges.
2. **Learning rate sweep** — α too small / just right / too large.
3. **Feature scaling impact** — convergence with vs. without z-score scaling at the same α.
4. **Predicted vs. actual** — scatter on the test set.

## Run the UI

```bash
streamlit run app.py
```

## Concepts demonstrated (Andrew Ng, Course 1, Week 1–2)

- Supervised learning: regression on a labelled dataset.
- Cost function: MSE formulated as `(1/2m) Σ (f(xᵢ) − yᵢ)²`.
- Gradient descent: simultaneous update of `w` and `b` with a learning rate.
- Vectorization: model, cost, and gradients are NumPy matrix operations.
- Feature scaling: z-score normalization; saved `μ` and `σ` to scale inputs at inference.
