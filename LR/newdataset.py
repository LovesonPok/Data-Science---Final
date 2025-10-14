# Purpose: Merge ecological (AvgTempC, WindSpeedMph) by date with cleaned dataset.
# - Creates date1 from cleaned time column (date only)
# - Merges with ecological Date column
# - Keeps all cleaned rows, repeats matches, leaves NaN if no match
# - Drops helper column date1 after merge

from __future__ import annotations
import pandas as pd
from pathlib import Path

# Paths (change if needed)
CLEANED = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
ECO_CSV = Path("/Users/lovesonpokhrel/Documents/Data Science/Ecological_factors.csv")
OUT_CSV = Path("/Users/lovesonpokhrel/Documents/Data Science/CsvForInvesB.csv")

# --- Load data ---
df = pd.read_csv(CLEANED)
eco = pd.read_csv(ECO_CSV)

# --- Identify and convert time/date columns ---
if "time" not in df.columns:
    raise SystemExit("Error: No 'time' column found in cleaned dataset.")
if "Date" not in eco.columns:
    raise SystemExit("Error: No 'Date' column found in ecological dataset.")

df["time"] = pd.to_datetime(df["time"], errors="coerce")
eco["Date"] = pd.to_datetime(eco["Date"], errors="coerce")

# --- Create date1 column (date only) for matching ---
df["date1"] = df["time"].dt.date

# --- Prepare ecological subset ---
eco_subset = eco[["Date", "AvgTempC", "WindSpeedMph"]].copy()
eco_subset["Date"] = eco_subset["Date"].dt.date

# --- Merge based on date ---
merged = pd.merge(
    df,
    eco_subset,
    how="left",
    left_on="date1",
    right_on="Date"
)

# --- Clean up ---
merged.drop(columns=["date1", "Date"], inplace=True)

# --- Save output ---
merged.to_csv(OUT_CSV, index=False)
print(f"Merged file saved at: {OUT_CSV}")
print("New columns added: AvgTempC, WindSpeedMph")
print(f"Non-null counts:\n{merged[['AvgTempC','WindSpeedMph']].notna().sum()}")
