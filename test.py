# investigation_b_lr.py
# Investigation B: Seasonal Linear Regression workflow (Weeks 1–9 aligned)

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

# Modeling + metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional diagnostics
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ====== CONFIG ======
# Use the exact output from your dataclean.py
CLEAN_PATH = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")

# Choose your response variable (continuous). You can change this later.
RESPONSE = "bat_landing_to_food"

# Candidate explanatory variables — we’ll auto-select from these if they exist
CANDIDATE_X = [
    # bat/rat context (dataset2)
    "bat_landing_number", "rat_minutes", "rat_arrival_number", "food_availability",
    # timing context (dataset1 + dataset2)
    "hours_after_sunset",
    # risk/reward behavior (kept as numeric if 0/1)
    "risk", "reward",
]

# Interaction terms to probe (optional; added if both parents exist)
INTERACTIONS = [
    ("rat_minutes", "hours_after_sunset"),
    ("rat_arrival_number", "hours_after_sunset"),
]

# File outputs
OUT_DIR = CLEAN_PATH.parent
METRICS_CSV = OUT_DIR / "IB_lr_metrics_by_season.csv"
COEFFS_CSV  = OUT_DIR / "IB_lr_coeffs_by_season.csv"
RESID_PNG_WINTER = OUT_DIR / "IB_residuals_winter.png"
RESID_PNG_SPRING = OUT_DIR / "IB_residuals_spring.png"
CORR_PNG          = OUT_DIR / "IB_corr_numeric.png"

# ====== UTILITIES ======
def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'season' column exists for ALL rows:
      0 if date < 2018-03-01 00:00:00, else 1
    Date source = start_time if present, else time, else leave as-is if season exists.
    """
    if "season" in df.columns:
        # If the season column exists, keep as is (already integers from your cleaner)
        return df
    date_col = None
    for c in ["start_time", "time"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # If we truly have no date info, default all to 1 (spring) to avoid NA issues.
        df["season"] = 1
        return df
    dt = safe_to_datetime(df[date_col])
    df["season"] = (dt >= pd.Timestamp("2018-03-01 00:00:00")).astype("int")
    return df

def add_interactions(df: pd.DataFrame, interactions: list[tuple[str,str]]) -> pd.DataFrame:
    for a, b in interactions:
        if a in df.columns and b in df.columns:
            name = f"{a}__x__{b}"
            df[name] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")
    return df

def select_features(df: pd.DataFrame, y_name: str, candidates: list[str]) -> list[str]:
    use = []
    for c in candidates:
        if c in df.columns:
            # keep only numeric for LR
            if pd.api.types.is_numeric_dtype(df[c]):
                use.append(c)
            else:
                # attempt numeric conversion (e.g., 0/1 strings)
                try_series = pd.to_numeric(df[c], errors="coerce")
                if try_series.notna().any():
                    df[c] = try_series
                    use.append(c)
    # Drop the response if it slipped into candidates
    use = [c for c in use if c != y_name]
    return use

def train_lr(X: pd.DataFrame, y: pd.Series) -> tuple[LinearRegression, dict]:
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    metrics = {
        "n": int(len(y)),
        "r2": float(r2_score(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
    }
    return model, metrics

def plot_residuals(y: pd.Series, yhat: np.ndarray, title: str, out_png: Path):
    resid = y - yhat
    # Residual vs Fitted
    plt.figure()
    plt.scatter(yhat, resid, s=14)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"{title}: Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(out_png)

def corr_plot_numeric(df: pd.DataFrame, out_png: Path):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        plt.figure()
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Numeric Correlations")
        plt.tight_layout()
        plt.savefig(out_png)

def compute_vif(dfX: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(dfX, has_constant="add")
    vifs = []
    for i, col in enumerate(X.columns):
        if col == "const": 
            continue
        vifs.append({"feature": col, "vif": float(variance_inflation_factor(X.values, i))})
    return pd.DataFrame(vifs).sort_values("vif", ascending=False)

# ====== MAIN ======
def main():
    df = pd.read_csv(CLEAN_PATH)
    # Basic time parsing if present
    for c in ["start_time", "time", "sunrise_time", "sunset_time"]:
        if c in df.columns:
            df[c] = safe_to_datetime(df[c])

    # Ensure 'season' exists exactly per rule (0 before 2018-03-01, else 1)
    df = ensure_season(df)

    # Make sure response exists and is numeric
    if RESPONSE not in df.columns:
        raise SystemExit(f"Response column '{RESPONSE}' not found in {CLEAN_PATH}")
    df[RESPONSE] = pd.to_numeric(df[RESPONSE], errors="coerce")

    # Feature selection + optional interactions
    X_cols = select_features(df, RESPONSE, CANDIDATE_X)
    df = add_interactions(df, INTERACTIONS)
    # add any created interactions that are numeric
    for col in list(df.columns):
        if "__x__" in col and pd.api.types.is_numeric_dtype(df[col]) and col not in X_cols:
            X_cols.append(col)

    # Drop rows with NA in y or any X
    base_cols = [RESPONSE] + X_cols + ["season"]
    dfm = df[base_cols].dropna().copy()

    # Quick correlation overview (Weeks 5–7)
    corr_plot_numeric(dfm, CORR_PNG)

    # Split by season as required (Week 9 regression practice)
    results_metrics = []
    coeff_rows = []

    for season_value, label, resid_png in [(0, "Winter (0)", RESID_PNG_WINTER),
                                           (1, "Spring (1)", RESID_PNG_SPRING)]:
        dsub = dfm[dfm["season"] == season_value].copy()
        if dsub.empty:
            continue

        y = dsub[RESPONSE]
        X = dsub[X_cols]
        # Train LR
        model, metrics = train_lr(X, y)

        # Save metrics
        metrics.update({"season": int(season_value), "label": label})
        results_metrics.append(metrics)

        # Coeff table
        coefs = dict(zip(X_cols, model.coef_))
        for k, v in coefs.items():
            coeff_rows.append({"season": int(season_value), "feature": k, "coef": float(v)})
        # residual plot
        yhat = model.predict(X)
        plot_residuals(y, yhat, f"Investigation B LR — {label}", resid_png)

        # Print console summary (Weeks 8–9 style)
        print(f"\n=== {label} ===")
        print(f"n={metrics['n']}  R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")
        # Top coefficients by magnitude
        top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        print("Top coefficients:")
        for name, val in top:
            print(f"  {name:30s} {val: .4f}")

        # Optional: VIF to check multicollinearity (Week 7)
        try:
            vif_df = compute_vif(X)
            print("\nVIF (top 10):")
            print(vif_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"(VIF skipped: {e})")

    # Save metrics & coefficients for report tables
    if results_metrics:
        pd.DataFrame(results_metrics).to_csv(METRICS_CSV, index=False)
    if coeff_rows:
        pd.DataFrame(coeff_rows).to_csv(COEFFS_CSV, index=False)

    print("\nSaved outputs:")
    print(f"  Metrics:   {METRICS_CSV}")
    print(f"  Coeffs:    {COEFFS_CSV}")
    print(f"  Residuals: {RESID_PNG_WINTER}, {RESID_PNG_SPRING}")
    print(f"  Corr img:  {CORR_PNG}")

if __name__ == "__main__":
    main()
