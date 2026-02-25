"""
COVID-19 ABM Dashboard - Intervention Scenario Comparison

Compare intervention strategies side-by-side to see health AND economic
tradeoffs. Helps decision-makers evaluate lockdown, masks, vaccination,
and combined policies before implementation.

Launch with:
    python dashboard.py
    Open http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from model import InfectionModel

# ── App setup ──────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="COVID-19 ABM - Scenario Comparison")

# ── Predefined scenarios ──────────────────────────────────────────────

SCENARIOS = {
    "No Intervention": {},
    "Lockdown Only": {"lockdown": True},
    "Masks + Awareness": {"ipa": True, "mm": True},
    "Vaccination Only": {"vaccination": True, "vaccination_rate": 0.02},
    "Full Response": {"lockdown": True, "saq": True, "ipa": True, "mm": True,
                      "vaccination": True, "vaccination_rate": 0.03},
}

SCENARIO_COLORS = {
    "No Intervention": "#e74c3c",
    "Lockdown Only": "#f39c12",
    "Masks + Awareness": "#3498db",
    "Vaccination Only": "#9b59b6",
    "Full Response": "#27ae60",
}

# ── Layout ─────────────────────────────────────────────────────────────

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COVID-19 ABM — Intervention Scenario Comparison",
                style={"margin": "0", "fontSize": "22px"}),
        html.P("Compare health and economic tradeoffs across intervention strategies",
               style={"margin": "0", "fontSize": "12px", "opacity": "0.7"}),
    ], style={"padding": "15px 20px", "backgroundColor": "#2c3e50", "color": "white"}),

    # Main content
    html.Div([
        # Left sidebar: Controls
        html.Div([
            html.H3("Shared Parameters", style={"marginTop": "0"}),

            html.Label("Population"),
            dcc.Slider(id="pop-slider", min=50, max=500, step=50, value=200,
                       marks={50: "50", 200: "200", 500: "500"}),

            html.Label("Transmission Probability"),
            dcc.Slider(id="ptrans-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                       marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2"}),

            html.Label("Initial Infected %"),
            dcc.Slider(id="init-inf-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                       marks={0.01: "1%", 0.1: "10%", 0.2: "20%"}),

            html.Label("Simulation Days"),
            dcc.Slider(id="days-slider", min=30, max=180, step=10, value=60,
                       marks={30: "30", 60: "60", 90: "90", 120: "120", 180: "180"}),

            html.Label("Runs per Scenario (for averaging)"),
            dcc.Slider(id="runs-slider", min=1, max=5, step=1, value=3,
                       marks={1: "1", 3: "3", 5: "5"}),

            html.Hr(),
            html.H3("Scenarios to Compare"),

            dcc.Checklist(
                id="scenario-checklist",
                options=[{"label": f"  {name}", "value": name} for name in SCENARIOS],
                value=list(SCENARIOS.keys()),
                style={"lineHeight": "2.2"},
            ),

            html.Hr(),

            html.Button("Run Comparison", id="run-btn", n_clicks=0,
                        style={"width": "100%", "padding": "12px", "fontSize": "15px",
                               "backgroundColor": "#27ae60", "color": "white",
                               "border": "none", "borderRadius": "6px", "cursor": "pointer",
                               "fontWeight": "bold"}),

            html.Div(id="status-text",
                     style={"marginTop": "12px", "fontSize": "13px", "color": "#666"}),

        ], style={"width": "280px", "padding": "15px", "backgroundColor": "#ffffff",
                  "borderRight": "1px solid #ddd", "overflowY": "auto"}),

        # Right content: Comparison charts
        html.Div([
            # Summary table
            html.Div(id="summary-table-container",
                     style={"marginBottom": "15px"}),

            # Row 1: Infection curves + Deaths
            html.Div([
                html.Div([
                    dcc.Graph(id="infection-plot", style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
                html.Div([
                    dcc.Graph(id="deaths-plot", style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),

            # Row 2: R0 + Wealth impact
            html.Div([
                html.Div([
                    dcc.Graph(id="r0-plot", style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
                html.Div([
                    dcc.Graph(id="wealth-plot", style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                      "marginTop": "10px"}),

        ], style={"flex": "1", "padding": "15px", "overflowY": "auto"}),

    ], style={"display": "flex", "height": "calc(100vh - 70px)"}),

], style={"fontFamily": "system-ui, -apple-system, sans-serif",
          "backgroundColor": "#f8f9fa", "height": "100vh", "overflow": "hidden"})


# ── Simulation runner ─────────────────────────────────────────────────

def run_scenario(name, scenario_params, shared_params, n_runs, max_steps):
    """Run a single scenario n_runs times and return averaged time series."""
    all_runs = []

    for run_idx in range(n_runs):
        params = {
            "N": shared_params["N"],
            "width": 15, "height": 15,
            "ptrans": shared_params["ptrans"],
            "initial_infected_perc": shared_params["initial_infected_perc"],
            "max_steps": max_steps,
            "seed": run_idx * 1000,
        }
        params.update(scenario_params)

        model = InfectionModel(**params)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        df["step"] = range(len(df))
        all_runs.append(df)

    # Average across runs
    combined = pd.concat(all_runs)
    averaged = combined.groupby("step").mean(numeric_only=True).reset_index()
    averaged["scenario"] = name
    return averaged


# ── Callbacks ──────────────────────────────────────────────────────────

@callback(
    Output("summary-table-container", "children"),
    Output("infection-plot", "figure"),
    Output("deaths-plot", "figure"),
    Output("r0-plot", "figure"),
    Output("wealth-plot", "figure"),
    Output("status-text", "children"),
    Input("run-btn", "n_clicks"),
    State("pop-slider", "value"),
    State("ptrans-slider", "value"),
    State("init-inf-slider", "value"),
    State("days-slider", "value"),
    State("runs-slider", "value"),
    State("scenario-checklist", "value"),
    prevent_initial_call=True,
)
def run_comparison(n_clicks, pop, ptrans, init_inf, max_steps, n_runs, selected_scenarios):
    if not selected_scenarios:
        return no_update, no_update, no_update, no_update, no_update, "Select at least one scenario."

    shared_params = {
        "N": pop,
        "ptrans": ptrans,
        "initial_infected_perc": init_inf,
    }

    # Run all selected scenarios
    all_results = []
    for name in selected_scenarios:
        scenario_params = SCENARIOS.get(name, {})
        result = run_scenario(name, scenario_params, shared_params, n_runs, max_steps)
        all_results.append(result)

    results_df = pd.concat(all_results, ignore_index=True)

    # ── Summary table ──
    summary_rows = []
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        final = sc.iloc[-1]
        peak_inf = sc["infected"].max()
        peak_inf_day = int(sc.loc[sc["infected"].idxmax(), "step"])

        summary_rows.append({
            "Scenario": name,
            "Peak Infected (%)": f"{peak_inf:.1f}",
            "Peak Day": str(peak_inf_day),
            "Total Deaths": f"{final['dead']:.1f}",
            "Peak R0": f"{sc['R0'].max():.2f}",
            "Final Vaccinated (%)": f"{final.get('vaccinated', 0):.1f}",
            "Final Wealth (Most Poor)": f"{final.get('Most Poor', 0):.0f}",
            "Final Wealth (Most Rich)": f"{final.get('Most Rich', 0):.0f}",
        })

    summary_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary_rows[0].keys()],
        data=summary_rows,
        style_header={"backgroundColor": "#2c3e50", "color": "white",
                      "fontWeight": "bold", "fontSize": "12px", "padding": "8px"},
        style_cell={"textAlign": "center", "padding": "8px", "fontSize": "12px",
                    "border": "1px solid #ddd"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )

    # ── Infection curves (overlaid) ──
    infection_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        color = SCENARIO_COLORS.get(name, "#999")
        infection_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["infected"], mode="lines",
            name=name, line=dict(color=color, width=2.5),
        ))
    infection_fig.update_layout(
        title="Infected Population (%) Over Time", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="% Infected",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── Deaths (overlaid) ──
    deaths_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        color = SCENARIO_COLORS.get(name, "#999")
        deaths_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["dead"], mode="lines",
            name=name, line=dict(color=color, width=2.5),
        ))
    deaths_fig.update_layout(
        title="Cumulative Deaths Over Time", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="Deaths",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── R0 (overlaid) ──
    r0_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        color = SCENARIO_COLORS.get(name, "#999")
        r0_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["R0"], mode="lines",
            name=name, line=dict(color=color, width=2.5),
        ))
    r0_fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="R0 = 1 (epidemic threshold)")
    r0_fig.update_layout(
        title="Reproduction Number (R0) Over Time", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="R0",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── Wealth impact (grouped bar: final wealth per stratum per scenario) ──
    wealth_fig = go.Figure()
    strata = ["Most Poor", "Poor", "Middle Class", "Rich", "Most Rich"]
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        final = sc.iloc[-1]
        values = [final.get(s, 0) for s in strata]
        color = SCENARIO_COLORS.get(name, "#999")
        wealth_fig.add_trace(go.Bar(
            x=strata, y=values, name=name,
            marker_color=color,
        ))
    wealth_fig.update_layout(
        title="Final Wealth by Social Stratum", title_x=0.5, title_font_size=14,
        xaxis_title="Social Stratum", yaxis_title="Total Wealth",
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=40, b=70),
    )

    total_runs = len(selected_scenarios) * n_runs
    status = f"Ran {len(selected_scenarios)} scenarios x {n_runs} runs = {total_runs} simulations ({max_steps} days each)."

    return summary_table, infection_fig, deaths_fig, r0_fig, wealth_fig, status


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting COVID-19 ABM Dashboard...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=False, port=8050)
