"""Preprocessing for the Pune (or Bengaluru) house dataset.

Expected raw columns (Kaggle "Pune House Data" / "Bengaluru House Data"):
    area_type, availability, site_location (or 'location'), size, society,
    total_sqft, bath, balcony, price

`price` is in lakhs of rupees. This module:
  1. Parses messy columns (size -> bhk, sqft ranges -> midpoint)
  2. Drops rows with critical nulls
  3. Removes geometric outliers (impossibly small per-BHK, extreme price/sqft)
  4. Caps locality cardinality (top N + "other")
  5. One-hot encodes locality and area_type
  6. Returns a clean numeric DataFrame ready for the model
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


CORE_NUMERIC = ["total_sqft", "bhk", "bath", "balcony"]


def _parse_sqft(value) -> float | None:
    """Handle '1133', '1133-1384', '1200Sq. Meter', etc."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Range like "1133 - 1384" -> midpoint
    if "-" in s:
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except ValueError:
            return None
    # Strip trailing non-numeric units (e.g., "Sq. Meter", "Acres"). For unit
    # conversion we'd want a full table; for simplicity we drop those rows.
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bhk(value) -> int | None:
    if pd.isna(value):
        return None
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Some versions of the dataset name the column 'location' instead of
    # 'site_location'. Normalize.
    if "site_location" not in df.columns and "location" in df.columns:
        df = df.rename(columns={"location": "site_location"})
    return df


def clean(
    df: pd.DataFrame,
    top_n_localities: int = 30,
    ppsqft_z_thresh: float = 3.0,
) -> pd.DataFrame:
    df = df.copy()

    # 1. Parse messy fields
    df["bhk"] = df["size"].apply(_parse_bhk)
    df["total_sqft"] = df["total_sqft"].apply(_parse_sqft)

    # 2. Keep only the columns we care about
    keep = ["area_type", "site_location", "total_sqft", "bath", "balcony", "bhk", "price"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # 3. Drop rows with critical nulls
    required = [c for c in ["total_sqft", "bath", "bhk", "price", "site_location"] if c in df.columns]
    df = df.dropna(subset=required)

    # 'balcony' missing values are common -> fill with 0
    if "balcony" in df.columns:
        df["balcony"] = df["balcony"].fillna(0)

    # Cast numerics
    for c in ["total_sqft", "bath", "balcony", "bhk", "price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)

    # 4. Geometric sanity: at least 300 sqft per BHK
    df = df[df["total_sqft"] / df["bhk"] >= 300]

    # Clean locality whitespace before grouping
    df["site_location"] = df["site_location"].astype(str).str.strip()

    # 5. Per-locality price-per-sqft z-score outlier removal (vectorized: build
    # a boolean mask using groupby transform — no apply, no column-drop pitfalls)
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]  # rupees per sqft
    grouped = df.groupby("site_location")["price_per_sqft"]
    loc_mean = grouped.transform("mean")
    loc_std = grouped.transform("std").fillna(0)
    keep_mask = (loc_std == 0) | (np.abs(df["price_per_sqft"] - loc_mean) <= ppsqft_z_thresh * loc_std)
    df = df[keep_mask].reset_index(drop=True)

    # 6. Cap locality cardinality
    counts = df["site_location"].value_counts()
    top = set(counts.head(top_n_localities).index)
    df["site_location"] = df["site_location"].where(df["site_location"].isin(top), other="other")

    df = df.drop(columns=["price_per_sqft"])
    return df.reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One-hot encode categoricals and return (X, y, feature_names).

    If `feature_columns` is supplied, columns are reindexed to match — useful
    for transforming a single user-input row at inference time so the column
    order is identical to training.
    """
    work = df.copy()

    categorical = [c for c in ["site_location", "area_type"] if c in work.columns]
    work = pd.get_dummies(work, columns=categorical, drop_first=False)

    y = work["price"].to_numpy(dtype=float) if "price" in work.columns else None
    X_df = work.drop(columns=["price"], errors="ignore")

    if feature_columns is not None:
        X_df = X_df.reindex(columns=feature_columns, fill_value=0)

    X = X_df.to_numpy(dtype=float)
    return X, y, list(X_df.columns)


def get_top_localities(df: pd.DataFrame) -> list[str]:
    return sorted(df["site_location"].unique().tolist())
