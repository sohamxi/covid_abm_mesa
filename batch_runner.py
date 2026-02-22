"""
Batch parameter sweep and sensitivity analysis for the COVID-19 ABM.

Uses Mesa's built-in batch_run for parallel parameter exploration
and provides sensitivity analysis via variance decomposition.
"""

import numpy as np
import pandas as pd
from itertools import product
from mesa import batch_run
from model import InfectionModel


# Default parameter ranges for sweeps
DEFAULT_PARAM_RANGES = {
    "N": [200],
    "width": [15],
    "height": [15],
    "ptrans": [0.02, 0.05, 0.08, 0.12],
    "initial_infected_perc": [0.02, 0.05, 0.10],
    "death_rate": [0.01, 0.02, 0.04],
    "severe_perc": [0.10, 0.18, 0.25],
    "lockdown": [False, True],
    "vaccination": [False, True],
    "vaccination_rate": [0.01, 0.03],
}


def run_parameter_sweep(param_ranges=None, iterations=3, max_steps=60, n_processes=1):
    """
    Run a full parameter sweep using Mesa's batch_run.

    Args:
        param_ranges: Dict of parameter name -> list of values to sweep
        iterations: Number of random seeds per parameter combo
        max_steps: Steps per simulation
        n_processes: Number of parallel processes (1 = sequential)

    Returns:
        pd.DataFrame with all results
    """
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES

    print(f"Starting parameter sweep...")
    combos = 1
    for vals in param_ranges.values():
        combos *= len(vals)
    print(f"  {combos} parameter combinations x {iterations} iterations = {combos * iterations} runs")

    results = batch_run(
        InfectionModel,
        parameters=param_ranges,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=n_processes,
        data_collection_period=1,
        display_progress=True,
    )

    df = pd.DataFrame(results)
    print(f"Sweep complete: {len(df)} data points collected")
    return df


def extract_summary_metrics(sweep_df):
    """
    Extract key summary metrics from batch run results.

    For each run, computes:
        - peak_infected: Maximum % infected at any step
        - total_dead: Final dead count
        - peak_R0: Maximum R0
        - time_to_peak: Step at which infection peaked
        - total_recovered: Final recovered count
    """
    summaries = []

    # Group by RunId
    for run_id, run_df in sweep_df.groupby("RunId"):
        final = run_df.iloc[-1]
        peak_row = run_df.loc[run_df["infected"].idxmax()] if "infected" in run_df else None

        summary = {
            "RunId": run_id,
            "ptrans": final.get("ptrans", None),
            "initial_infected_perc": final.get("initial_infected_perc", None),
            "death_rate": final.get("death_rate", None),
            "severe_perc": final.get("severe_perc", None),
            "lockdown": final.get("lockdown", None),
            "vaccination": final.get("vaccination", None),
            "vaccination_rate": final.get("vaccination_rate", None),
            "peak_infected": run_df["infected"].max() if "infected" in run_df else 0,
            "total_dead": final.get("dead", 0),
            "peak_R0": run_df["R0"].max() if "R0" in run_df else 0,
            "mean_R0": run_df["R0"].mean() if "R0" in run_df else 0,
            "time_to_peak": peak_row["Step"] if peak_row is not None else 0,
            "total_recovered": final.get("recovered", 0),
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


def sensitivity_analysis(sweep_df, output_var="peak_infected"):
    """
    One-at-a-time sensitivity analysis using variance decomposition.

    For each parameter, computes how much of the output variance it explains.

    Args:
        sweep_df: DataFrame from extract_summary_metrics()
        output_var: Which output metric to analyze

    Returns:
        pd.DataFrame with columns: parameter, variance_explained, mean_effect
    """
    results = []
    total_var = sweep_df[output_var].var()

    if total_var == 0:
        print("Warning: zero variance in output - all runs produced same result")
        return pd.DataFrame(columns=["parameter", "variance_explained_pct", "mean_effect"])

    params_to_analyze = ["ptrans", "initial_infected_perc", "death_rate", "severe_perc",
                         "lockdown", "vaccination", "vaccination_rate"]

    for param in params_to_analyze:
        if param not in sweep_df.columns:
            continue
        if sweep_df[param].nunique() <= 1:
            continue

        group_means = sweep_df.groupby(param)[output_var].mean()
        between_var = group_means.var() * len(group_means)
        var_explained = between_var / total_var * 100

        # Mean effect: difference between max and min group means
        mean_effect = group_means.max() - group_means.min()

        results.append({
            "parameter": param,
            "variance_explained_pct": round(var_explained, 1),
            "mean_effect": round(mean_effect, 2),
            "values_tested": list(sweep_df[param].unique()),
        })

    result_df = pd.DataFrame(results).sort_values("variance_explained_pct", ascending=False)
    return result_df


def run_scenario_comparison(scenarios, n_runs=5, max_steps=60):
    """
    Compare named intervention scenarios.

    Args:
        scenarios: Dict of scenario_name -> dict of model params
        n_runs: Number of runs per scenario
        max_steps: Steps per run

    Returns:
        pd.DataFrame with scenario results
    """
    all_results = []

    for name, params in scenarios.items():
        print(f"Running scenario: {name}")
        for run_idx in range(n_runs):
            m = InfectionModel(seed=run_idx * 1000, **params)
            for _ in range(max_steps):
                m.step()

            df = m.datacollector.get_model_vars_dataframe()
            df["scenario"] = name
            df["run"] = run_idx
            df["step"] = range(len(df))
            all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# Predefined scenarios for comparison
SCENARIOS = {
    "No Intervention": {
        "N": 200, "width": 15, "height": 15, "ptrans": 0.05,
    },
    "Lockdown Only": {
        "N": 200, "width": 15, "height": 15, "ptrans": 0.05,
        "lockdown": True,
    },
    "Masks + Awareness": {
        "N": 200, "width": 15, "height": 15, "ptrans": 0.05,
        "ipa": True, "mm": True,
    },
    "Vaccination Only": {
        "N": 200, "width": 15, "height": 15, "ptrans": 0.05,
        "vaccination": True, "vaccination_rate": 0.02,
    },
    "Full Response": {
        "N": 200, "width": 15, "height": 15, "ptrans": 0.05,
        "lockdown": True, "saq": True, "ipa": True, "mm": True,
        "vaccination": True, "vaccination_rate": 0.03,
    },
}


if __name__ == "__main__":
    # Quick demo
    print("=" * 60)
    print("Running scenario comparison...")
    print("=" * 60)
    results = run_scenario_comparison(SCENARIOS, n_runs=3, max_steps=60)

    # Print final stats per scenario
    final_steps = results.groupby(["scenario", "run"]).last().groupby("scenario").mean(numeric_only=True)
    print("\n" + "=" * 60)
    print("Mean final state by scenario (averaged over 3 runs):")
    print("=" * 60)
    for scenario in SCENARIOS:
        row = final_steps.loc[scenario]
        print(f"\n{scenario}:")
        print(f"  Infected: {row['infected']:.1f}%, Dead: {row['dead']:.1f}")
        print(f"  Recovered: {row['recovered']:.1f}%, R0: {row['R0']:.2f}")
        if "vaccinated" in row:
            print(f"  Vaccinated: {row['vaccinated']:.1f}%")
