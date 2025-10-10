
# Investigation A – Interaction test and alternate target
# Model A (Integrated + Interaction): 
#   y = bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season
#                              + bat_landing_number + rat_arrival_number + food_availability
#                              + rat_minutes + season:rat_minutes
# Model B 
#   y = bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season
#                              + rat_arrival_number + food_availability + rat_minutes
#                              + season:rat_minutes

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------
# Paths / Load
# ---------------------------
DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
FIG_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/Figures")
OUT_DIR = Path("/Users/lovesonpokhrel/Documents/Data Science/Outputs")
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

def metrics(name: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2  = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    f2   = float(r2 / (1 - r2 + 1e-12))
    print(f"\n[{name}]  n={len(y_true)}  R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  Cohen's f²={f2:.3f}")
    return {"n": len(y_true), "R2": r2, "RMSE": rmse, "MAE": mae, "f2": f2}

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

# ---------------------------
# Model A — Integrated + Interaction (season × rat_minutes)
# ---------------------------
cols_A = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "bat_landing_number", "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_A = [c for c in cols_A if c not in df.columns]
if missing_A:
    raise SystemExit(f"Missing required columns for Model A: {missing_A}")

A_df = to_num(df[cols_A].copy(), cols_A).dropna()

print("\n=== MODEL A (Integrated + Interaction):")
print("bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season "
      "+ bat_landing_number + rat_arrival_number + food_availability + rat_minutes + season:rat_minutes")

formula_A = (
    "bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "bat_landing_number + rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)

modA = smf.ols(formula_A, data=A_df).fit()
print(modA.summary())
print("\n95% Confidence Intervals (Model A):\n",
      modA.conf_int().rename(columns={0:"ci_low",1:"ci_high"}))

mA = metrics("Model A (Integrated + Interaction)", modA.model.endog, modA.fittedvalues)

# Save outputs A
coefA = modA.params.to_frame("coef")
coefA["std_err"] = modA.bse
coefA["t"] = modA.tvalues
coefA["p"] = modA.pvalues
ciA = modA.conf_int()
coefA["ci_low"] = ciA[0]; coefA["ci_high"] = ciA[1]
coefA.to_csv(OUT_DIR / "ModelA_coefficients.csv")

predA = A_df.copy()
predA["y_actual"] = modA.model.endog
predA["y_pred"] = modA.fittedvalues
predA["residual"] = modA.resid
predA.to_csv(OUT_DIR / "ModelA_predictions.csv", index=False)

save_heatmap(A_df, "Heatmap – Model A (Integrated + Interaction)", FIG_DIR / "ModelA_heatmap.png")

# Key scatterplots (A)
plt.figure(figsize=(8,6))
sns.regplot(data=A_df, x="rat_minutes", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model A: Bat landing to food vs Rat minutes"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelA_scatter_ratminutes_vs_batlandingtofood.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=A_df, x="food_availability", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model A: Bat landing to food vs Food availability"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelA_scatter_food_vs_batlandingtofood.png", dpi=300); plt.show()

diagnostics(modA.resid, modA.fittedvalues, "Model A", FIG_DIR / "ModelA_diagnostics.png")

# ---------------------------
# Model B — Alternate target (bat_landing_number) with same predictors 
# ---------------------------
cols_B = [
    "bat_landing_number",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_B = [c for c in cols_B if c not in df.columns]
if missing_B:
    raise SystemExit(f" Missing required columns for Model B: {missing_B}")

B_df = to_num(df[cols_B].copy(), cols_B).dropna()

print("\n=== MODEL B (Alt target, same Xs; season × rat_minutes interaction):")
print("bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season "
      "+ rat_arrival_number + food_availability + rat_minutes + season:rat_minutes")

formula_B = (
    "bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)

modB = smf.ols(formula_B, data=B_df).fit()
print(modB.summary())
print("\n95% Confidence Intervals (Model B):\n",
      modB.conf_int().rename(columns={0:"ci_low",1:"ci_high"}))

mB = metrics("Model B (Alt target + Interaction)", modB.model.endog, modB.fittedvalues)

# Save outputs B
coefB = modB.params.to_frame("coef")
coefB["std_err"] = modB.bse
coefB["t"] = modB.tvalues
coefB["p"] = modB.pvalues
ciB = modB.conf_int()
coefB["ci_low"] = ciB[0]; coefB["ci_high"] = ciB[1]
coefB.to_csv(OUT_DIR / "ModelB_coefficients.csv")

predB = B_df.copy()
predB["y_actual"] = modB.model.endog
predB["y_pred"] = modB.fittedvalues
predB["residual"] = modB.resid
predB.to_csv(OUT_DIR / "ModelB_predictions.csv", index=False)

save_heatmap(B_df, "Heatmap – Model B (Alt target + Interaction)", FIG_DIR / "ModelB_heatmap.png")

# Key scatterplots (B)
plt.figure(figsize=(8,6))
sns.regplot(data=B_df, x="rat_minutes", y="bat_landing_number",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model B: Bat landing number vs Rat minutes"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelB_scatter_ratminutes_vs_batlandingnumber.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=B_df, x="food_availability", y="bat_landing_number",
            scatter_kws={"alpha":0.45, "s":35}, line_kws={"color":"red","lw":2})
plt.title("Model B: Bat landing number vs Food availability"); plt.tight_layout()
plt.savefig(FIG_DIR / "ModelB_scatter_food_vs_batlandingnumber.png", dpi=300); plt.show()

diagnostics(modB.resid, modB.fittedvalues, "Model B", FIG_DIR / "ModelB_diagnostics.png")



print("\n Finished Model A (Integrated + season×rat_minutes) and Model B (Alt target with same Xs).")
print(f"Figures saved to: {FIG_DIR}")
print(f"Tables/Predictions saved to")
