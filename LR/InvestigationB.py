# Investigation B – Season-split models with ecological factors (AvgTempC, WindSpeedMph)
# Linear regression, train/test evaluation, 10-row Actual vs Predicted tables,
# and four visuals per season. Heatmaps issue solved,.

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Paths / Load
# ---------------------------
DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/CsvForInvesB.csv")
FIG_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/FiguresC")
OUT_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/OutputsC")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
sns.set(style="whitegrid", context="talk")

# ---------------------------
# Helpers
# ---------------------------
def to_num(df_: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df_.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def nrmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = float(np.max(y_true) - np.min(y_true))
    return rmse / denom if denom != 0 else np.nan

def print_fit_metrics(title: str, y, yhat, adj_r2: float | None = None):
    r2  = r2_score(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae  = float(mean_absolute_error(y, yhat))
    nrm  = nrmse(y, yhat)
    if adj_r2 is None:
        print(f"[{title}]  n={len(y)}  R²={r2:.3f}  RMSE={rmse:.3f}  NRMSE={nrm:.3f}  MAE={mae:.3f}")
    else:
        print(f"[{title}]  n={len(y)}  R²={r2:.3f}  Adj.R²={adj_r2:.3f}  RMSE={rmse:.3f}  NRMSE={nrm:.3f}  MAE={mae:.3f}")
    return {"n": len(y), "R2": r2, "AdjR2": adj_r2, "RMSE": rmse, "NRMSE": nrm, "MAE": mae}

def _strip_and_dedupe(cols: list[str]) -> list[str]:
    """Strip whitespace and dedupe duplicate names deterministically."""
    base = [str(c).strip() for c in cols]
    seen = {}
    out = []
    for c in base:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")  
    return out

def _drop_constant_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    """Drop numeric columns with zero variance (std == 0)."""
    if df_num.empty:
        return df_num
    std = df_num.std(numeric_only=True, ddof=0)
    keep_cols = std[std > 0].index.tolist()
    dropped = [c for c in df_num.columns if c not in keep_cols]
    if dropped:
        print(f"(Heatmap) Dropping constant columns (zero variance): {dropped}")
    return df_num[keep_cols]

def save_heatmap(
    df_sub: pd.DataFrame,
    title: str,
    path: Path,
    var_order: list[str] | None = None
) -> None:
    
    # numeric subset + clean names
    num = df_sub.select_dtypes(include="number").copy()
    if num.shape[1] < 2:
        print(f"(Skip heatmap: not enough numeric columns for {title})")
        return

    num.columns = _strip_and_dedupe(list(num.columns))
    num = _drop_constant_numeric(num)

    if num.shape[1] < 2:
        print(f"(Skip heatmap: not enough numeric columns after dropping constants for {title})")
        return

    # Build correlation using numpy to avoid edge-case differences
    cols_num = list(num.columns)
    corr_mat = np.corrcoef(num.values, rowvar=False)
    corr = pd.DataFrame(corr_mat, index=cols_num, columns=cols_num)

    # Consistent order for both axes
    if var_order:
        order = [c for c in var_order if c in corr.columns]
        remainder = [c for c in corr.columns if c not in order]
        order += remainder
    else:
        order = list(corr.columns)

    corr = corr.loc[order, order]

    print(f"\n=== Correlation Matrix: {title} ===")
    print(corr.round(3))
    print(f"(Debug) corr shape: {corr.shape}  labels: {len(order)}×{len(order)}")

    # Plot 
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(16, 14))
        hm = sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            center=0,
            annot=True, fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.6},
            xticklabels=order,   
            yticklabels=order
        )

        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=14, pad=10)
        ax.tick_params(axis="x", labelrotation=60, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0, labelsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=500)
        plt.show()

def diagnostics(resid, fitted, title_prefix: str, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=fitted, y=resid, ax=ax[0], alpha=0.6)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_title(f"{title_prefix}: Residuals vs Fitted")
    ax[0].set_xlabel("Fitted values"); ax[0].set_ylabel("Residuals")
    sm.qqplot(resid, line="45", fit=True, ax=ax[1])
    ax[1].set_title(f"{title_prefix}: Q–Q Plot")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

# ---------------------------
# Columns and data prep
# ---------------------------
cols = [
    "bat_landing_number", "seconds_after_rat_arrival", "bat_landing_to_food",
    "hours_after_sunset_d1", "risk", "season", "rat_arrival_number",
    "food_availability", "rat_minutes", "AvgTempC", "WindSpeedMph"
]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df = to_num(df, cols).dropna(subset=cols)

# Seasonal splits
df0 = df[df["season"] == 0].copy()
df1 = df[df["season"] == 1].copy()

VAR_ORDER = [
    "season", "bat_landing_to_food", "seconds_after_rat_arrival", "risk",
    "hours_after_sunset_d1", "bat_landing_number", "food_availability",
    "rat_minutes", "rat_arrival_number", "AvgTempC", "WindSpeedMph"
]

# ---------------------------
# Per-season modeling
# ---------------------------
def run_season_model(data: pd.DataFrame, season_label: str):
    if data.empty:
        print(f"\n=== MODEL – SEASON {season_label} ===")
        print("No rows for this season after cleaning. Skipping.")
        return

    print(f"\n=== MODEL – SEASON {season_label} ===")

    # 80/20 split 
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    formula = (
        "bat_landing_number ~ seconds_after_rat_arrival + bat_landing_to_food + "
        "hours_after_sunset_d1 + risk + rat_arrival_number + food_availability + "
        "rat_minutes + AvgTempC + WindSpeedMph + rat_minutes:AvgTempC"
    )

    model = smf.ols(formula, data=train).fit()
    print(model.summary())

    # Metrics
    _ = print_fit_metrics(
        f"Season {season_label} – Train",
        model.model.endog, model.fittedvalues, adj_r2=model.rsquared_adj
    )
    test_pred = model.predict(test)
    _ = print_fit_metrics(
        f"Season {season_label} – Test",
        test["bat_landing_number"], test_pred
    )

    # 10-row Actual vs Predicted
    comp = pd.DataFrame({
        "Actual": test["bat_landing_number"].values,
        "Predicted": test_pred.values
    }).reset_index(drop=True).head(10)
    print(f"\n--- Season {season_label}: Actual vs Predicted (first 10 test rows) ---")
    print(comp)
    comp.to_csv(OUT_DIR / f"Season{season_label}_Actual_vs_Predicted_head10.csv", index=False)

    # ---- Plots (4 per model) ----
    save_heatmap(
        data,
        f"Heatmap – Season {season_label}",
        FIG_DIR / f"Season{season_label}_heatmap.png",
        var_order=VAR_ORDER
    )

    plt.figure(figsize=(8,6))
    sns.regplot(
        data=data, x="rat_minutes", y="bat_landing_number",
        scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2}
    )
    plt.title(f"Season {season_label}: Bat landing number vs Rat minutes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"Season{season_label}_scatter_ratminutes.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,6))
    sns.regplot(
        data=data, x="AvgTempC", y="bat_landing_number",
        scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"blue","lw":2}
    )
    plt.title(f"Season {season_label}: Bat landing number vs Temperature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"Season{season_label}_scatter_temp.png", dpi=300)
    plt.show()

    diagnostics(
        model.resid, model.fittedvalues,
        f"Season {season_label}",
        FIG_DIR / f"Season{season_label}_diagnostics.png"
    )

# ---------------------------
# Run both season models
# ---------------------------
run_season_model(df0, "0")
run_season_model(df1, "1")

print("\nInvestigation B complete.")
print(f"Figures saved to: {FIG_DIR}")
print(f"Outputs saved to: {OUT_DIR}")
