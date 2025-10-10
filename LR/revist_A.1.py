
# Final Investigation A (cleaned) — ONLY two models kept as discussed with team in chat and meeting.
# Model 1 (Integrated + Interaction)
#   y = bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season
#                              + bat_landing_number + rat_arrival_number + food_availability
#                              + rat_minutes + season:rat_minutes
#
# Model 2 (Alt target)
#   y = bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season
#                              + rat_arrival_number + food_availability + rat_minutes
#                              + season:rat_minutes
#
# Visuals saved for BOTH models:
#   - Heatmap (correlations)
#   - Pairplot (pairwise relationships)
#   - Key scatterplots
#   - Diagnostics: Residuals vs Fitted, Q–Q plot
# Tables saved:
#   - Coefficients (with CI)
#   - Predictions (actual, fitted, residual)
#   - Metrics summary

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
DATA   = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
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
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    f2   = float(r2 / (1 - r2 + 1e-12))
    print(f"\n[{name}]  n={len(y_true)}  R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  Cohen's f²={f2:.3f}")
    return {"n": len(y_true), "R2": r2, "RMSE": rmse, "MAE": mae, "f2": f2}

def save_heatmap(df_sub: pd.DataFrame, title: str, path: Path) -> None:
    num = df_sub.select_dtypes(include="number")
    if num.shape[1] < 2:
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

# ===========================
# Model 1 — Integrated + Interaction (season × rat_minutes)
# ===========================
cols_m1 = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "bat_landing_number", "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_m1 = [c for c in cols_m1 if c not in df.columns]
if missing_m1:
    raise SystemExit(f" Missing required columns for Model 1: {missing_m1}")

m1_df = to_num(df[cols_m1].copy(), cols_m1).dropna()

print("\n=== MODEL 1 (Integrated + Interaction) ===")
formula_m1 = (
    "bat_landing_to_food ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "bat_landing_number + rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)
m1 = smf.ols(formula_m1, data=m1_df).fit()
print(m1.summary())
print("\n95% Confidence Intervals (Model 1):\n",
      m1.conf_int().rename(columns={0:"ci_low",1:"ci_high"}))
m1_metrics = metrics("Model 1", m1.model.endog, m1.fittedvalues)

# Save coefficients & predictions
coef1 = m1.params.to_frame("coef")
coef1["std_err"], coef1["t"], coef1["p"] = m1.bse, m1.tvalues, m1.pvalues
ci1 = m1.conf_int(); coef1["ci_low"], coef1["ci_high"] = ci1[0], ci1[1]
coef1.to_csv(OUT_DIR / "Model1_coefficients.csv")

pred1 = m1_df.copy()
pred1["y_actual"], pred1["y_pred"], pred1["residual"] = m1.model.endog, m1.fittedvalues, m1.resid
pred1.to_csv(OUT_DIR / "Model1_predictions.csv", index=False)

# Visuals — Model 1
save_heatmap(m1_df, "Heatmap – Model 1 (Integrated + Interaction)", FIG_DIR / "Model1_heatmap.png")

sns.pairplot(m1_df, diag_kind="kde", corner=True)
plt.suptitle("Pairwise Relationships – Model 1", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "Model1_pairplot.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=m1_df, x="rat_minutes", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45,"s":35}, line_kws={"color":"red","lw":2})
plt.title("Model 1: Bat landing to food vs Rat minutes")
plt.tight_layout(); plt.savefig(FIG_DIR / "Model1_scatter_ratminutes_vs_batlandingtofood.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=m1_df, x="food_availability", y="bat_landing_to_food",
            scatter_kws={"alpha":0.45,"s":35}, line_kws={"color":"red","lw":2})
plt.title("Model 1: Bat landing to food vs Food availability")
plt.tight_layout(); plt.savefig(FIG_DIR / "Model1_scatter_food_vs_batlandingtofood.png", dpi=300); plt.show()

diagnostics(m1.resid, m1.fittedvalues, "Model 1", FIG_DIR / "Model1_diagnostics.png")

# ===========================
# Model 2 — Alternate target ()
# ===========================
cols_m2 = [
    "bat_landing_number",
    "seconds_after_rat_arrival", "hours_after_sunset_d1", "risk", "season",
    "rat_arrival_number", "food_availability", "rat_minutes"
]
missing_m2 = [c for c in cols_m2 if c not in df.columns]
if missing_m2:
    raise SystemExit(f"Missing required columns for Model 2: {missing_m2}")

m2_df = to_num(df[cols_m2].copy(), cols_m2).dropna()

print("\n=== MODEL 2 (Alt target + Interaction) ===")
formula_m2 = (
    "bat_landing_number ~ seconds_after_rat_arrival + hours_after_sunset_d1 + risk + season + "
    "rat_arrival_number + food_availability + rat_minutes + season:rat_minutes"
)
m2 = smf.ols(formula_m2, data=m2_df).fit()
print(m2.summary())
print("\n95% Confidence Intervals (Model 2):\n",
      m2.conf_int().rename(columns={0:"ci_low",1:"ci_high"}))
m2_metrics = metrics("Model 2", m2.model.endog, m2.fittedvalues)

# Save coefficients & predictions
coef2 = m2.params.to_frame("coef")
coef2["std_err"], coef2["t"], coef2["p"] = m2.bse, m2.tvalues, m2.pvalues
ci2 = m2.conf_int(); coef2["ci_low"], coef2["ci_high"] = ci2[0], ci2[1]
coef2.to_csv(OUT_DIR / "Model2_coefficients.csv")

pred2 = m2_df.copy()
pred2["y_actual"], pred2["y_pred"], pred2["residual"] = m2.model.endog, m2.fittedvalues, m2.resid
pred2.to_csv(OUT_DIR / "Model2_predictions.csv", index=False)

# Visuals — Model 2
save_heatmap(m2_df, "Heatmap – Model 2 (Alt target + Interaction)", FIG_DIR / "Model2_heatmap.png")

sns.pairplot(m2_df, diag_kind="kde", corner=True)
plt.suptitle("Pairwise Relationships – Model 2", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "Model2_pairplot.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=m2_df, x="rat_minutes", y="bat_landing_number",
            scatter_kws={"alpha":0.45,"s":35}, line_kws={"color":"red","lw":2})
plt.title("Model 2: Bat landing number vs Rat minutes")
plt.tight_layout(); plt.savefig(FIG_DIR / "Model2_scatter_ratminutes_vs_batlandingnumber.png", dpi=300); plt.show()

plt.figure(figsize=(8,6))
sns.regplot(data=m2_df, x="food_availability", y="bat_landing_number",
            scatter_kws={"alpha":0.45,"s":35}, line_kws={"color":"red","lw":2})
plt.title("Model 2: Bat landing number vs Food availability")
plt.tight_layout(); plt.savefig(FIG_DIR / "Model2_scatter_food_vs_batlandingnumber.png", dpi=300); plt.show()

diagnostics(m2.resid, m2.fittedvalues, "Model 2", FIG_DIR / "Model2_diagnostics.png")

print("\nFinished Model 1 and Model 2. Figures saved to:", FIG_DIR)
print("Tables/Predictions saved to:")
