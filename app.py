"""Streamlit UI for the Pune house price predictor.

Run:
    streamlit run app.py

Loads the trained from-scratch model + scaler + feature columns and the cleaned
dataset. Takes user preferences in the sidebar, predicts a price, and surfaces
matching listings sorted by 'underpriced for the area' (listed - predicted).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.linear_regression import LinearRegressionGD
from src.logistic_regression import LogisticRegressionGD
from src.preprocess import build_feature_matrix
from src.scaler import ZScoreScaler


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "pune_clean.csv"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"

# Palette (colorhunt: fbfbfb / e8f9ff / c4d9ff / c5baff)
PALETTE = {
    "bg": "#FBFBFB",
    "surface": "#E8F9FF",
    "accent": "#C4D9FF",
    "primary": "#C5BAFF",
    "ink": "#1A1A2E",
    "muted": "#5C5C73",
    # Semantic deal colors (kept soft to fit the pastel palette)
    "good_bg": "#D9F5E1", "good_ink": "#1E6B3A",
    "fair_bg": "#FFF4D6", "fair_ink": "#7A5B00",
    "over_bg": "#FFE0E0", "over_ink": "#8A1E1E",
}

SIDEBAR_CSS = """
<style>
/* Dark-blue sidebar: lighten text and form labels for legibility */
[data-testid="stSidebar"] { background-color: #578FCA; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

/* Inputs inside the sidebar: keep their internal field readable (dark text on light field) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] * { color: #1A1A2E !important; }

/* Predict button: white pill with dark text on the blue sidebar.
   The inner descendants need their own override because the earlier
   sidebar `*` rule sets every descendant's color to white. */
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF !important;
    color: #1A1A2E !important;
    border: 1px solid #FFFFFF !important;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {
    color: #1A1A2E !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #E8F9FF !important;
    border-color: #E8F9FF !important;
}

/* Main-area selectbox: match the card aesthetic (light bg, soft border).
   Only style the outermost visible pill — touching deeper BaseWeb layers
   breaks the dropdown toggle and the arrow click target. */
[data-testid="stMain"] [data-baseweb="select"] > div:first-child {
    background-color: #FBFBFB !important;
    border: 1px solid #C4D9FF !important;
    border-radius: 10px !important;
}
[data-testid="stMain"] [data-baseweb="select"] input {
    color: #1A1A2E !important;
    caret-color: transparent !important;  /* hide typing cursor */
}
[data-testid="stMain"] [data-testid="stSelectbox"] label {
    color: #5C5C73 !important;
    font-size: 0.8rem !important;
}
</style>
"""

CARD_CSS = f"""
<style>
.listing-card {{
    background: {PALETTE['bg']};
    border: 1px solid {PALETTE['accent']};
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(26,26,46,0.04);
    font-family: inherit;
}}
.listing-card .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-bottom: 10px;
}}
.listing-card .title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: {PALETTE['ink']};
    margin-bottom: 2px;
}}
.listing-card .meta {{
    font-size: 0.85rem;
    color: {PALETTE['muted']};
    margin-bottom: 12px;
}}
.listing-card .price-row {{
    display: flex; justify-content: space-between;
    font-size: 0.92rem; color: {PALETTE['ink']};
    padding: 3px 0;
}}
.listing-card .price-row .label {{ color: {PALETTE['muted']}; }}
.listing-card hr {{
    border: none; border-top: 1px dashed {PALETTE['accent']};
    margin: 10px 0 8px 0;
}}
.listing-card .delta {{
    font-size: 0.92rem; font-weight: 600;
}}
</style>
"""


def _accuracy_tier(listed: float, predicted: float) -> tuple[str, str, str]:
    """Return (label, bg_color, ink_color) based on prediction accuracy.

    Accuracy = how close the model's predicted price is to the listed price.
    Small |delta| -> the model fits this listing well.
    """
    if predicted <= 0:
        return "LOW CONFIDENCE", PALETTE["over_bg"], PALETTE["over_ink"]
    abs_pct = abs(listed - predicted) / predicted * 100
    if abs_pct <= 5:
        return "ACCURATE PREDICTION", PALETTE["good_bg"], PALETTE["good_ink"]
    if abs_pct <= 15:
        return "CLOSE FIT", PALETTE["fair_bg"], PALETTE["fair_ink"]
    return "LARGE GAP", PALETTE["over_bg"], PALETTE["over_ink"]


def render_listing_cards(ranked: pd.DataFrame) -> None:
    """Render the 2-column card grid.

    If the ranked frame contains a `clf_label` column (0/1), each card also
    gets a second pill: 'EXPENSIVE ≥ ₹Xl' or 'AFFORDABLE < ₹Xl'.
    The expected `clf_threshold` (in lakhs) lives in the optional `clf_threshold`
    column — same value per row, but easier than threading it through args.
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    rows = ranked.to_dict("records")
    # Render in a 2-column grid
    for i in range(0, len(rows), 2):
        cols = st.columns(2, gap="medium")
        for col, row in zip(cols, rows[i : i + 2]):
            listed = float(row["price"])
            predicted = float(row["predicted_price"])
            delta = listed - predicted
            label, bg, ink = _accuracy_tier(listed, predicted)
            pct = (delta / predicted * 100) if predicted else 0.0
            direction = "below" if delta < 0 else "above"
            bath = int(row.get("bath", 0))
            balcony = int(row.get("balcony", 0))
            area_type = row.get("area_type", "")

            # Optional classifier pill
            clf_html = ""
            if "clf_label" in row and pd.notna(row["clf_label"]):
                threshold = float(row.get("clf_threshold", 100.0))
                if int(row["clf_label"]) == 1:
                    clf_bg, clf_ink = PALETTE["accent"], PALETTE["ink"]
                    clf_text = f"EXPENSIVE ≥ ₹{threshold:.0f}L"
                else:
                    clf_bg, clf_ink = PALETTE["surface"], PALETTE["ink"]
                    clf_text = f"AFFORDABLE < ₹{threshold:.0f}L"
                clf_html = (
                    f'<span class="badge" style="background:{clf_bg}; '
                    f'color:{clf_ink}; margin-left:6px;">{clf_text}</span>'
                )

            with col:
                st.markdown(
                    f"""
<div class="listing-card">
  <span class="badge" style="background:{bg}; color:{ink};">{label}</span>{clf_html}
  <div class="title">{row['site_location']} · {int(row['bhk'])} BHK</div>
  <div class="meta">{int(row['total_sqft'])} sqft · {bath} bath · {balcony} balcony{f" · {area_type}" if area_type else ""}</div>
  <div class="price-row"><span class="label">Listed</span><span>₹ {listed:.1f} L</span></div>
  <div class="price-row"><span class="label">Model</span><span>₹ {predicted:.1f} L</span></div>
  <hr/>
  <div class="delta" style="color:{ink};">Δ {delta:+.1f} L ({abs(pct):.1f}% {direction} model)</div>
</div>
""",
                    unsafe_allow_html=True,
                )


@st.cache_resource
def load_artifacts():
    model = LinearRegressionGD.load(str(MODELS_DIR / "weights.npz"))
    scaler = ZScoreScaler.load(str(MODELS_DIR / "scaler.npz"))
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    df = pd.read_csv(PROCESSED_CSV)

    # Classifier is optional — the app degrades gracefully if it hasn't been trained.
    classifier = None
    clf_scaler = None
    clf_meta = None
    clf_weights_path = MODELS_DIR / "classifier_weights.npz"
    clf_scaler_path = MODELS_DIR / "classifier_scaler.npz"
    clf_meta_path = MODELS_DIR / "classifier_meta.json"
    if clf_weights_path.exists() and clf_scaler_path.exists() and clf_meta_path.exists():
        classifier = LogisticRegressionGD.load(str(clf_weights_path))
        clf_scaler = ZScoreScaler.load(str(clf_scaler_path))
        with open(clf_meta_path) as f:
            clf_meta = json.load(f)

    return model, scaler, feature_columns, df, classifier, clf_scaler, clf_meta


def classify_rows(
    df_or_row: pd.DataFrame, classifier, clf_scaler, feature_columns, clf_meta
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, predicted labels 0/1) for each row."""
    X, _, _ = build_feature_matrix(df_or_row, feature_columns=feature_columns)
    X_s = clf_scaler.transform(X)
    proba = classifier.predict_proba(X_s)
    label = (proba >= clf_meta.get("decision_threshold", 0.5)).astype(int)
    return proba, label


def predict_price(model, scaler, feature_columns, user_row: dict) -> float:
    """Build a single-row DataFrame that mirrors the training schema, then predict."""
    df = pd.DataFrame([user_row])
    X, _, _ = build_feature_matrix(df, feature_columns=feature_columns)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def main():
    st.set_page_config(page_title="Pune House Price Predictor", layout="wide")
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.title("Pune House Price Predictor")
    st.caption(
        "Predicting Pune house prices with linear regression built from scratch in NumPy — "
        "supervised learning, cost function, gradient descent, vectorization, and feature scaling."
    )

    try:
        model, scaler, feature_columns, df, classifier, clf_scaler, clf_meta = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Train the model first:\n\n"
            "    python -m src.train --csv data/raw/pune_houses.csv"
        )
        return

    classifier_ready = classifier is not None and clf_scaler is not None and clf_meta is not None

    localities = sorted(df["site_location"].unique().tolist())
    area_types = sorted(df["area_type"].unique().tolist()) if "area_type" in df.columns else []

    # ----- Sidebar inputs -----
    with st.sidebar:
        st.header("Your preferences")
        locality = st.selectbox("Locality", localities, index=0)
        bhk = st.slider("BHK", 1, 6, 2)
        bath = st.slider("Bathrooms", 1, 6, 2)
        balcony = st.slider("Balconies", 0, 4, 1)
        total_sqft = st.number_input("Total sqft", min_value=200, max_value=10000, value=1000, step=50)
        area_type = st.selectbox("Area type", area_types) if area_types else None
        predict_clicked = st.button("Predict price", type="primary", use_container_width=True)

    # ----- Prediction -----
    user_row = {
        "site_location": locality,
        "total_sqft": float(total_sqft),
        "bath": float(bath),
        "balcony": float(balcony),
        "bhk": float(bhk),
    }
    if area_type is not None:
        user_row["area_type"] = area_type

    if predict_clicked or True:  # always show a prediction
        predicted = predict_price(model, scaler, feature_columns, user_row)
        if classifier_ready:
            cols = st.columns(4)
            user_proba, user_label = classify_rows(
                pd.DataFrame([user_row]), classifier, clf_scaler, feature_columns, clf_meta
            )
            threshold = clf_meta["price_threshold"]
            verdict = "Expensive" if int(user_label[0]) == 1 else "Affordable"
            cols[3].metric(
                f"Class (₹{threshold:.0f}L cutoff)",
                verdict,
                f"p = {float(user_proba[0]):.2f}",
            )
        else:
            cols = st.columns(3)
        cols[0].metric("Predicted price", f"₹ {predicted:.1f} L")
        cols[1].metric("Per sqft", f"₹ {predicted * 1e5 / total_sqft:,.0f}")
        cols[2].metric(
            "Locality avg (sqft)",
            f"₹ {df.loc[df['site_location'] == locality, 'price'].mean() * 1e5 / df.loc[df['site_location'] == locality, 'total_sqft'].mean():,.0f}"
            if (df['site_location'] == locality).any() else "—",
        )

    # ----- Matching listings -----
    header_col, filter_col = st.columns([3, 1], gap="medium")
    with header_col:
        st.subheader("Matching listings from the dataset")
        st.caption("Filtered by your locality and BHK (±1). 'Delta' is listed − model-predicted, so negative values mean the listing is underpriced relative to the model.")

    candidates = df[
        (df["site_location"] == locality)
        & (df["bhk"].between(bhk - 1, bhk + 1))
    ].copy()

    if candidates.empty:
        st.info("No listings match this locality + BHK range in the dataset.")
    else:
        # Predict for every candidate row (regressor)
        X_cand, _, _ = build_feature_matrix(candidates, feature_columns=feature_columns)
        X_cand_s = scaler.transform(X_cand)
        candidates["predicted_price"] = model.predict(X_cand_s)
        candidates["delta"] = candidates["price"] - candidates["predicted_price"]

        # Ground-truth Expensive/Affordable label for every candidate row.
        # We use the *actual listed price* here (not the classifier's prediction)
        # because the cards display the listed price right next to the badge,
        # so a feature-based prediction looks contradictory when the listing is
        # an outlier for its locality. The classifier still earns its keep in
        # the top prediction panel, where the user's hypothetical house has no
        # known price.
        if classifier_ready:
            threshold = clf_meta["price_threshold"]
            candidates["clf_label"] = (candidates["price"] > threshold).astype(int)
            candidates["clf_threshold"] = threshold

        # Drop listings where the model is wildly off — these are almost
        # always dataset noise (typos, unusual sales) rather than real deals,
        # and surfacing them as "great bargains" would be misleading.
        candidates["delta_pct"] = candidates["delta"] / candidates["predicted_price"]
        trusted = candidates[candidates["delta_pct"].abs() <= 0.40].copy()

        # Soft relevance filter: prefer listings within ±40% of requested sqft
        # so a 600-sqft studio doesn't outrank a 1000-sqft match just because
        # it's underpriced.
        trusted = trusted[(trusted["total_sqft"] >= total_sqft * 0.6) & (trusted["total_sqft"] <= total_sqft * 1.4)]
        if trusted.empty:
            trusted = candidates  # fall back to unfiltered if nothing passes

        # Sort by smallest absolute delta first — the listings where the
        # model agrees most closely with the listed price (i.e. where the
        # prediction is most trustworthy) bubble to the top.
        trusted["abs_delta"] = trusted["delta"].abs()
        trusted["abs_pct"] = (trusted["abs_delta"] / trusted["predicted_price"].abs()) * 100
        trusted["accuracy_tier"] = pd.cut(
            trusted["abs_pct"],
            bins=[-0.01, 5, 15, float("inf")],
            labels=["ACCURATE PREDICTION", "CLOSE FIT", "LARGE GAP"],
        ).astype(str)

        tier_counts = trusted["accuracy_tier"].value_counts()
        tier_options = ["All accuracy tiers"] + [
            f"{tier} ({tier_counts.get(tier, 0)})"
            for tier in ["ACCURATE PREDICTION", "CLOSE FIT", "LARGE GAP"]
        ]
        with filter_col:
            selected = st.selectbox(
                "Filter by accuracy",
                tier_options,
                index=0,
                label_visibility="collapsed",
            )

        if not selected.startswith("All"):
            chosen_tier = selected.split(" (")[0]
            filtered = trusted[trusted["accuracy_tier"] == chosen_tier]
        else:
            filtered = trusted

        ranked = filtered.sort_values("abs_delta", ascending=True).head(10)

        if ranked.empty:
            st.info("No listings in this accuracy tier for your locality + BHK.")
        else:
            render_listing_cards(ranked)

    # ----- How this works -----
    with st.expander("How this works (the math behind the prediction)"):
        st.markdown(
            """
            **Model:** `f(x) = w · x + b` — linear regression.

            **Training:** batch gradient descent on the MSE cost
            `J(w, b) = (1/2m) Σ (f(xᵢ) − yᵢ)²`, with z-score feature scaling
            applied first so all features have comparable magnitude
            (sqft is ~1000s, BHK is ~2, one-hot columns are 0/1).

            **From scratch:** the model, the cost function, the gradient
            computation, the parameter updates, and the scaler are all
            implemented in NumPy. scikit-learn is used only as a baseline to
            verify the implementation is correct.
            """
        )
        fig = FIG_DIR / "1_cost_convergence.png"
        if fig.exists():
            st.image(str(fig), caption="Cost J(w,b) decreasing across gradient descent iterations")


if __name__ == "__main__":
    main()
