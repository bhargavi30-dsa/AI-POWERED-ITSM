import pandas as pd

df = pd.read_csv(
     r"D:\College_Projects\Enterprise_ITSM_AI\data\raw\itsm_cleaned_dataset.csv"
)

feature_values = {

    "contact_type": sorted(df["contact_type"].dropna().unique().tolist()),

    "location": sorted(df["location"].dropna().unique().tolist()),

    "u_symptom": sorted(df["u_symptom"].dropna().unique().tolist()),

    "impact": sorted(df["impact"].dropna().unique().tolist()),

    "urgency": sorted(df["urgency"].dropna().unique().tolist()),

    "notify": sorted(df["notify"].dropna().unique().tolist()),

    "opened_day_of_week": sorted(df["opened_day_of_week"].dropna().unique().tolist()),

    "opened_month": sorted(df["opened_month"].dropna().unique().tolist())
}