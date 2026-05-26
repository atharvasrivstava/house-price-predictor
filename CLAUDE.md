# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

A Pune house-price predictor whose **explicit goal is to demonstrate Andrew Ng ML Specialization Course 1 (Week 1–2) concepts** — cost function, batch gradient descent, vectorization, and z-score feature scaling — implemented from scratch in NumPy. scikit-learn is intentionally restricted to the role of a *baseline* (`src/evaluate.py`) so the from-scratch implementation can be verified for parity. Do not refactor the from-scratch model in `src/linear_regression.py` or the scaler in `src/scaler.py` to delegate to library implementations; that would defeat the project's purpose.

## Common commands

All commands run from the project root. A virtualenv (`.venv/`) is expected.

```bash
# Dependencies
pip install -r requirements.txt

# Train the from-scratch regressor (produces models/weights.npz, models/scaler.npz,
# models/feature_columns.json, and data/processed/pune_clean.csv)
python -m src.train --csv data/raw/pune_houses.csv

# Train the from-scratch classifier (Expensive vs Affordable).
# MUST run AFTER src.train — depends on the cleaned CSV and feature_columns.json.
python -m src.train_classifier

# Verify the regressor against sklearn.LinearRegression
python -m src.evaluate --csv data/raw/pune_houses.csv

# Verify the classifier against sklearn.LogisticRegression
python -m src.evaluate_classifier

# Generate the four diagnostic plots into notebooks/figures/
python -m src.diagnostics --csv data/raw/pune_houses.csv

# Run the Streamlit UI (consumes the trained artifacts in models/)
streamlit run app.py
```

There is no test suite or linter configured. The de facto correctness check is `python -m src.evaluate` — the from-scratch model's MAE/RMSE/R² should match sklearn's `LinearRegression` within a small margin.

## Architecture

### The training → inference contract

Three artifacts written by `src/train.py` form a strict contract that both `src/evaluate.py` and `app.py` depend on:

- `models/weights.npz` — trained `w` (n,) and `b` (scalar) for the linear model
- `models/scaler.npz` — `mu` and `sigma` per feature for z-score normalization
- `models/feature_columns.json` — **ordered** list of feature names

`feature_columns.json` is load-bearing. When `app.py` builds a single-row feature matrix for a user's input via `preprocess.build_feature_matrix(df, feature_columns=...)`, that argument forces the one-hot-encoded columns into the *exact same order* the model was trained on. Any change to the preprocessing schema (new categorical, dropped column, etc.) requires retraining so all three artifacts stay in sync.

### Data flow

1. `src/preprocess.load_raw` reads the Kaggle CSV and renames `location` → `site_location` if needed (the well-known "Bengaluru House Data" schema uses `location`; this project supports both).
2. `src/preprocess.clean` parses messy fields (`"2 BHK"` → `bhk`, `"1133-1384"` → midpoint), drops geometric outliers (`sqft/bhk < 300`), removes per-locality price-per-sqft outliers via a vectorized `groupby().transform()` mask (not `groupby().apply()` — `apply` with `include_groups=False` strips the grouping column and caused a real bug previously), and caps locality cardinality to a top-N plus `"other"`.
3. `src/preprocess.build_feature_matrix` one-hot encodes `site_location` and `area_type`, then reindexes to `feature_columns` if provided.
4. `src/scaler.ZScoreScaler` applies `(X - mu) / sigma`, with a guard that replaces zero σ with 1 to avoid divide-by-zero on one-hot columns that happened to be all-zero in a split.
5. `src/linear_regression.LinearRegressionGD` performs batch gradient descent. `fit` populates `cost_history` for use by the convergence diagnostics; `predict` is `X @ w + b`. No Python loops over training examples — gradients are computed via `X.T @ err / m`.

### Streamlit UI specifics (`app.py`)

- The user's input is converted to a one-row DataFrame and passed through the **same** `build_feature_matrix(df, feature_columns=...)` and `scaler.transform(...)` pipeline as training. This is why the saved `feature_columns.json` matters.
- The matching-listings section applies two filters before sorting:
  - **Trust filter:** drop candidates where `|delta| / predicted > 40%` (these are almost always dataset noise — typos or distressed sales — and surfacing them as "deals" would mislead users).
  - **Soft relevance filter:** keep only listings within ±40% of requested sqft.
  - Falls back to unfiltered candidates if both filters wipe the result set.
- Listings are sorted by **smallest absolute delta first** so the model's most-confident predictions appear at the top. The accuracy badge on each card (`ACCURATE PREDICTION` / `CLOSE FIT` / `LARGE GAP`) reflects this same `|delta| / predicted` percentage, not deal direction.
- `.streamlit/config.toml` defines the light theme. `app.py` injects two CSS blocks: `SIDEBAR_CSS` (dark-blue sidebar with white text overrides — note the `[data-testid="stSidebar"] .stButton > button *` override that exists specifically to defeat the broad `[data-testid="stSidebar"] *` color rule) and `CARD_CSS` (card styling). When restyling Streamlit widgets, **do not touch deep BaseWeb internals** (`[data-baseweb="select"] > div > div`) — doing so previously broke the selectbox arrow toggle. Only style the outermost visible layer (`[data-baseweb="select"] > div:first-child`).

### Logistic regression classifier (Course 1, Week 3)

A second from-scratch model in [src/logistic_regression.py](src/logistic_regression.py) classifies listings as **Expensive (price > price_threshold lakhs)** vs **Affordable**. It mirrors the linear regressor's structure: sigmoid, vectorized BCE cost with L2, batch GD, no Python loops over examples. Bias `b` is **not** regularized — only `w` is.

Three additional artifacts in `models/`:

- `classifier_weights.npz` — `w`, `b`
- `classifier_scaler.npz` — `mu`, `sigma` (separate instance; the classifier owns its own scaler to avoid coupling, even though the feature space is identical to the regressor's)
- `classifier_meta.json` — `{price_threshold, decision_threshold, lam, alpha, iters}`

The classifier intentionally reuses `models/feature_columns.json` from the regressor — both models share the exact same feature space. **Training order matters:** if you change preprocessing, retrain the regressor *before* the classifier so both pick up the new feature columns.

`app.py` loads classifier artifacts in `load_artifacts()` but treats them as **optional** — the app degrades gracefully (skips classification UI, no error) if they aren't present. Each card in the matching-listings grid renders a second pill (`EXPENSIVE ≥ ₹100L` / `AFFORDABLE < ₹100L`) using the existing `.badge` CSS rule with inline colors from `PALETTE['accent']` / `PALETTE['surface']` — do not add new badge classes.

### Deployment

The app is deployed on Streamlit Community Cloud at https://house-price-predictor-t.streamlit.app. The runtime expects model artifacts to be committed to the repo (they are small, total ~12K), so a fresh checkout can `streamlit run app.py` without retraining.
