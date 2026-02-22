"""
Real-world COVID-19 data loading from Our World in Data (OWID).

Provides functions to:
  - Download and cache OWID COVID-19 data
  - Extract country-specific time series
  - Compute normalized metrics for comparison with ABM output
  - Generate calibration targets

Data source: https://github.com/owid/covid-19-data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# OWID CSV URL (full dataset)
OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "owid_covid_data.csv")


def download_owid_data(force=False):
    """
    Download OWID COVID-19 dataset and cache locally.

    Args:
        force: If True, re-download even if cache exists

    Returns:
        pd.DataFrame with full OWID dataset
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force and os.path.exists(CACHE_FILE):
        # Use cache if less than 7 days old
        mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if (datetime.now() - mod_time).days < 7:
            return pd.read_csv(CACHE_FILE, parse_dates=["date"])

    try:
        print(f"Downloading OWID COVID-19 data...")
        df = pd.read_csv(OWID_URL, parse_dates=["date"])
        df.to_csv(CACHE_FILE, index=False)
        print(f"Downloaded {len(df)} rows, cached to {CACHE_FILE}")
        return df
    except Exception as e:
        print(f"Failed to download OWID data: {e}")
        if os.path.exists(CACHE_FILE):
            print("Using cached data instead.")
            return pd.read_csv(CACHE_FILE, parse_dates=["date"])
        raise


def get_country_data(country_code="USA", start_date=None, end_date=None, max_days=60):
    """
    Get COVID-19 time series for a specific country.

    Args:
        country_code: ISO country code (e.g., "USA", "BRA", "GBR", "IND")
        start_date: Start date string "YYYY-MM-DD" (default: first case date)
        end_date: End date string (default: start + max_days)
        max_days: Maximum number of days to return

    Returns:
        pd.DataFrame with columns:
            day, total_cases_per_million, new_cases_per_million,
            total_deaths_per_million, new_deaths_per_million,
            icu_patients_per_million, hosp_patients_per_million,
            people_vaccinated_per_hundred, stringency_index
    """
    df = download_owid_data()

    country_df = df[df["iso_code"] == country_code].copy()
    if country_df.empty:
        raise ValueError(f"No data found for country code: {country_code}")

    country_df = country_df.sort_values("date")

    if start_date:
        country_df = country_df[country_df["date"] >= start_date]
    else:
        # Start from first case
        first_case = country_df[country_df["total_cases"] > 0]["date"].min()
        if pd.notna(first_case):
            country_df = country_df[country_df["date"] >= first_case]

    if end_date:
        country_df = country_df[country_df["date"] <= end_date]
    elif max_days:
        country_df = country_df.head(max_days)

    # Create day index (0-based)
    if not country_df.empty:
        country_df["day"] = range(len(country_df))

    columns = {
        "day": "day",
        "date": "date",
        "total_cases_per_million": "total_cases_per_million",
        "new_cases_per_million": "new_cases_per_million",
        "total_deaths_per_million": "total_deaths_per_million",
        "new_deaths_per_million": "new_deaths_per_million",
        "icu_patients_per_million": "icu_patients_per_million",
        "hosp_patients_per_million": "hosp_patients_per_million",
        "people_vaccinated_per_hundred": "people_vaccinated_per_hundred",
        "stringency_index": "stringency_index",
        "reproduction_rate": "reproduction_rate",
    }

    result = pd.DataFrame()
    for new_col, old_col in columns.items():
        if old_col in country_df.columns:
            result[new_col] = country_df[old_col].values
        else:
            result[new_col] = np.nan

    return result


def normalize_for_comparison(real_data, model_data, population):
    """
    Normalize ABM output to match OWID per-million scale for comparison.

    Args:
        real_data: DataFrame from get_country_data()
        model_data: DataFrame from model.datacollector.get_model_vars_dataframe()
        population: Number of agents in the ABM

    Returns:
        Dict with 'real' and 'model' DataFrames aligned for plotting
    """
    model_norm = pd.DataFrame()
    scale = 1_000_000 / population  # Scale agent counts to per-million

    if "infected" in model_data.columns:
        model_norm["active_cases_per_million"] = model_data["infected"] * population / 100 * scale
    if "dead" in model_data.columns:
        model_norm["total_deaths_per_million"] = model_data["dead"] * scale
    if "recovered" in model_data.columns:
        model_norm["recovered_per_million"] = model_data["recovered"] * population / 100 * scale
    if "R0" in model_data.columns:
        model_norm["reproduction_rate"] = model_data["R0"]

    model_norm["day"] = range(len(model_norm))

    return {
        "real": real_data,
        "model": model_norm,
    }


def get_calibration_targets(country_code="USA", wave="first"):
    """
    Extract key calibration targets from real data.

    Returns dict of target metrics to match during parameter tuning:
        - peak_infected_day: Day of peak active cases
        - peak_infected_per_million: Peak active cases per million
        - total_deaths_per_million_60d: Total deaths per million at day 60
        - peak_R0: Maximum reproduction rate observed
    """
    if wave == "first":
        # First wave: early 2020
        data = get_country_data(country_code, start_date="2020-03-01", max_days=90)
    elif wave == "delta":
        data = get_country_data(country_code, start_date="2021-07-01", max_days=90)
    elif wave == "omicron":
        data = get_country_data(country_code, start_date="2021-12-01", max_days=90)
    else:
        data = get_country_data(country_code, max_days=90)

    targets = {}

    cases = data["new_cases_per_million"].dropna()
    if not cases.empty:
        targets["peak_new_cases_day"] = int(cases.idxmax()) if not cases.empty else None
        targets["peak_new_cases_per_million"] = float(cases.max())

    deaths = data["total_deaths_per_million"].dropna()
    if not deaths.empty and len(deaths) >= 60:
        targets["total_deaths_per_million_60d"] = float(deaths.iloc[59])

    r0 = data["reproduction_rate"].dropna()
    if not r0.empty:
        targets["peak_R0"] = float(r0.max())
        targets["mean_R0"] = float(r0.mean())

    return targets
