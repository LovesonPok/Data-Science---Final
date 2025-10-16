# Investigation A – Interaction test and alternate target
# All the test for LR learned from week 8 and 9,

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

# Paths / Load
DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
FIG_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/Figures")
OUT_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/Outputs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
sns.set(style="whitegrid", context="talk")

# Helpers
def to_num(df_: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df_.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def nrmse(y_true, y_pred):
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

def save_heatmap(df_sub: pd.DataFrame, title: str, path: Path) -> None:
    num = df_sub.select_dtypes(include="number")
    if num.shape[1] < 2 or num.empty:
        print(f"(Skip heatmap: not enough numeric columns for {title})")
        return
    corr = num.corr(numeric_only=True)
    print(f"\n=== Correlation Matrix: {title} ===\n{corr.round(3)}\n")
    plt.figure(figsize=(9,7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=300); plt.show()

def diagnostics(resid, fitted, title_prefix: str, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=fitted, y=resid, ax=ax[0], alpha=0.6)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_title(f"{title_prefix}: Residuals vs Fitted")
    ax[0].set_xlabel("Fitted values"); ax[0].set_ylabel("Residuals")
    sm.qqplot(resid, line="45", fit=True, ax=ax[1])
    ax[1].set_title(f"{title_prefix}: Q–Q Plot")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.show()

# Model A — Integrated + Interaction (season × rat_minutes)
cols_A = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "bat_landing_number", "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_A = [c for c in cols_A if c not in df.columns]
if missing_A:
    raise SystemExit(f"Missing required columns for Model A: {missing_A}")

A_df_full = to_num(df[cols_A].copy(), cols_A).dropna()

# 80/20 split
A_train, A_test = train_test_split(A_df_full, test_size=0.2, random_state=42)

formula_A = (
    "bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "bat_landing_number + rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)

print("\n=== MODEL A (Integrated + Interaction):")
modA = smf.ols(formula_A, data=A_train).fit()
print(modA.summary())

# Training metrics (use model's adj R²)
_ = print_fit_metrics("Model A – Train", modA.model.endog, modA.fittedvalues, adj_r2=modA.rsquared_adj)

# Test predictions + metrics
A_test_pred = modA.predict(A_test)
_ = print_fit_metrics("Model A – Test", A_test["bat_landing_to_food"], A_test_pred, adj_r2=None)

# Show & save 10-row Actual vs Predicted table (test set)
A_comp = pd.DataFrame({
    "Actual": A_test["bat_landing_to_food"].values,
    "Predicted": A_test_pred.values
}).reset_index(drop=True).head(10)
print("\n--- Model A: Actual vs Predicted (first 10 from test set) ---")
print(A_comp)
A_comp.to_csv(OUT_DIR / "ModelA_test_Actual_vs_Predicted_head10.csv", index=False)

# Save extra outputs/plots as before (on full data for visuals)
save_heatmap(A_df_full, "Heatmap – Model A (Integrated + Interaction)", FIG_DIR / "ModelA_heatmap.png")

plt.figure(figsize=(8,6))
sns.regplot(data=A_df_full, x="rat_minutes", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model A: Bat landing to food vs Rat minutes"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelA_scatter_ratminutes_vs_batlandingtofood.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=A_df_full, x="food_availability", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model A: Bat landing to food vs Food availability"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelA_scatter_food_vs_batlandingtofood.png", dpi=300); plt.show()

diagnostics(modA.resid, modA.fittedvalues, "Model A (train fit)", FIG_DIR / "ModelA_diagnostics.png")

# Model B — Alternate target (bat_landing_number) with same predictors 
cols_B = [
    "bat_landing_number",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_B = [c for c in cols_B if c not in df.columns]
if missing_B:
    raise SystemExit(f"Missing required columns for Model B: {missing_B}")

B_df_full = to_num(df[cols_B].copy(), cols_B).dropna()

# 80/20 split
B_train, B_test = train_test_split(B_df_full, test_size=0.2, random_state=42)

formula_B = (
    "bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)

print("\n=== MODEL B (Alt target, same Xs; season × rat_minutes):")
modB = smf.ols(formula_B, data=B_train).fit()
print(modB.summary())

# Training metrics
_ = print_fit_metrics("Model B – Train", modB.model.endog, modB.fittedvalues, adj_r2=modB.rsquared_adj)

# Test predictions + metrics
B_test_pred = modB.predict(B_test)
_ = print_fit_metrics("Model B – Test", B_test["bat_landing_number"], B_test_pred, adj_r2=None)

# 10-row Actual vs Predicted table (test set)
B_comp = pd.DataFrame({
    "Actual": B_test["bat_landing_number"].values,
    "Predicted": B_test_pred.values
}).reset_index(drop=True).head(10)
print("\n--- Model B: Actual vs Predicted (first 10 from test set) ---")
print(B_comp)
B_comp.to_csv(OUT_DIR / "ModelB_test_Actual_vs_Predicted_head10.csv", index=False)

# Plots on full data (for visuals)
save_heatmap(B_df_full, "Heatmap – Model B (Alt target + Interaction)", FIG_DIR / "ModelB_heatmap.png")

plt.figure(figsize=(8,6))
sns.regplot(data=B_df_full, x="rat_minutes", y="bat_landing_number",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model B: Bat landing number vs Rat minutes"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelB_scatter_ratminutes_vs_batlandingnumber.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=B_df_full, x="food_availability", y="bat_landing_number",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model B: Bat landing number vs Food availability"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelB_scatter_food_vs_batlandingnumber.png", dpi=300); plt.show()

diagnostics(modB.resid, modB.fittedvalues, "Model B (train fit)", FIG_DIR / "ModelB_diagnostics.png")

print("\nDone. NRMSE added and 10-row Actual vs Predicted tables printed & saved.")
print(f"Figures -> {FIG_DIR}")
print(f"Tables  -> {OUT_DIR}")
