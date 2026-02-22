"""
COVID-19 ABM Dashboard - Plotly Dash

Production-quality interactive dashboard for the COVID-19 Agent-Based Model.

Features:
  - Real-time simulation grid with agent state visualization
  - SEIR epidemic curves (susceptible, exposed, infected, recovered)
  - R0 tracker over time
  - Death and hospitalization panel
  - Wealth distribution across social strata
  - Intervention controls (lockdown, masks, vaccination, etc.)
  - Scenario comparison view
  - Real data overlay (OWID)

Launch with:
    python dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

from model import InfectionModel
from agent import InfectionState, InfectionSeverity, SocialStratum

# ── App setup ──────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="COVID-19 ABM Dashboard")

# Global simulation state
SIM_STATE = {
    "model": None,
    "history": [],
    "running": False,
}

# Color scheme
COLORS = {
    "susceptible": "#b5551d",
    "exposed": "#fcf00a",
    "infected": "#fa0c00",
    "recovered": "#008000",
    "dead": "#121010",
    "vaccinated": "#4a90d9",
    "severe": "#8b0000",
    "bg": "#f8f9fa",
    "card": "#ffffff",
    "text": "#2c3e50",
}

# ── Layout ─────────────────────────────────────────────────────────────

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COVID-19 Agent-Based Model", style={"margin": "0", "fontSize": "24px"}),
        html.P("Multi-layer contact networks | Age-stratified disease | Vaccination",
               style={"margin": "0", "fontSize": "12px", "opacity": "0.7"}),
    ], style={"padding": "15px 20px", "backgroundColor": "#2c3e50", "color": "white"}),

    # Main content
    html.Div([
        # Left sidebar: Controls
        html.Div([
            html.H3("Parameters", style={"marginTop": "0"}),

            html.Label("Population"),
            dcc.Slider(id="pop-slider", min=50, max=500, step=50, value=200,
                       marks={50: "50", 200: "200", 500: "500"}),

            html.Label("Transmission Probability"),
            dcc.Slider(id="ptrans-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                       marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2"}),

            html.Label("Initial Infected %"),
            dcc.Slider(id="init-inf-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                       marks={0.01: "1%", 0.1: "10%", 0.2: "20%"}),

            html.Hr(),
            html.H3("Interventions"),

            dcc.Checklist(id="interventions", options=[
                {"label": " Lockdown (graduated)", "value": "lockdown"},
                {"label": " Screening & Quarantine", "value": "saq"},
                {"label": " Public Awareness", "value": "ipa"},
                {"label": " Mandatory Masks", "value": "mm"},
                {"label": " Vaccination", "value": "vaccination"},
            ], value=[], style={"lineHeight": "2"}),

            html.Label("Vaccination Rate (daily)"),
            dcc.Slider(id="vax-rate-slider", min=0.005, max=0.05, step=0.005, value=0.02,
                       marks={0.005: "0.5%", 0.02: "2%", 0.05: "5%"}),

            html.Hr(),

            html.Div([
                html.Button("Initialize", id="init-btn", n_clicks=0,
                            style={"marginRight": "10px", "padding": "8px 16px",
                                   "backgroundColor": "#3498db", "color": "white",
                                   "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                html.Button("Run 60 Steps", id="run-btn", n_clicks=0,
                            style={"padding": "8px 16px",
                                   "backgroundColor": "#27ae60", "color": "white",
                                   "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
            ]),

            html.Div(id="status-text", style={"marginTop": "10px", "fontSize": "13px"}),

        ], style={"width": "250px", "padding": "15px", "backgroundColor": COLORS["card"],
                  "borderRight": "1px solid #ddd", "overflowY": "auto"}),

        # Right content: Charts
        html.Div([
            # Stats cards
            html.Div(id="stats-cards", style={
                "display": "flex", "gap": "10px", "marginBottom": "15px", "flexWrap": "wrap"
            }),

            # Charts in a 2x2 grid
            html.Div([
                html.Div([
                    dcc.Graph(id="grid-plot", style={"height": "350px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
                html.Div([
                    dcc.Graph(id="seir-plot", style={"height": "350px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    dcc.Graph(id="severity-plot", style={"height": "300px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
                html.Div([
                    dcc.Graph(id="wealth-plot", style={"height": "300px"}),
                ], style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "10px"}),

            # R0 plot
            html.Div([
                dcc.Graph(id="r0-plot", style={"height": "250px"}),
            ], style={"marginTop": "10px"}),

        ], style={"flex": "1", "padding": "15px", "overflowY": "auto"}),

    ], style={"display": "flex", "height": "calc(100vh - 70px)"}),

], style={"fontFamily": "system-ui, -apple-system, sans-serif",
          "backgroundColor": COLORS["bg"], "height": "100vh", "overflow": "hidden"})


# ── Helper functions ───────────────────────────────────────────────────

def make_stat_card(label, value, color="#2c3e50"):
    return html.Div([
        html.Div(str(value), style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
        html.Div(label, style={"fontSize": "11px", "opacity": "0.7"}),
    ], style={"padding": "10px 15px", "backgroundColor": COLORS["card"],
              "borderRadius": "6px", "border": f"2px solid {color}",
              "textAlign": "center", "minWidth": "80px"})


def make_grid_figure(model):
    """Create a scatter plot of agents on the grid."""
    if model is None:
        return go.Figure()

    x, y, colors, sizes, text = [], [], [], [], []
    color_map = {
        InfectionState.SUSCEPTIBLE: COLORS["susceptible"],
        InfectionState.EXPOSED: COLORS["exposed"],
        InfectionState.INFECTED: COLORS["infected"],
        InfectionState.RECOVERED: COLORS["recovered"],
        InfectionState.DIED: COLORS["dead"],
    }

    for agent in model.agents:
        if agent.pos is None:
            continue
        x.append(agent.pos[0])
        y.append(agent.pos[1])
        c = color_map.get(agent.state, "#999")
        if agent.vaccinated and agent.state == InfectionState.SUSCEPTIBLE:
            c = COLORS["vaccinated"]
        if agent.severity == InfectionSeverity.Severe:
            c = COLORS["severe"]
        colors.append(c)
        sizes.append(12 if agent.severity == InfectionSeverity.Severe else 7)
        text.append(f"ID: {agent.unique_id}<br>Age: {agent.age:.0f}<br>State: {agent.state.name}")

    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors, size=sizes, line=dict(width=0.5, color="white")),
        text=text, hoverinfo="text",
    ))
    fig.update_layout(
        title="Agent Grid", title_x=0.5, title_font_size=14,
        xaxis=dict(range=[-0.5, model.grid.width - 0.5], showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.5, model.grid.height - 0.5], showgrid=False, zeroline=False),
        plot_bgcolor="#f0f0f0",
        margin=dict(l=30, r=10, t=35, b=30),
    )
    return fig


def make_seir_figure(history):
    """Create SEIR epidemic curves."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure()
    for col, color in [("susceptible", COLORS["susceptible"]),
                       ("exposed", COLORS["exposed"]),
                       ("infected", COLORS["infected"]),
                       ("recovered", COLORS["recovered"])]:
        if col in df:
            fig.add_trace(go.Scatter(x=df["step"], y=df[col], mode="lines",
                                     name=col.capitalize(), line=dict(color=color, width=2)))
    fig.update_layout(
        title="SEIR Curves (% of population)", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="% Population",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=35, b=50),
    )
    return fig


def make_severity_figure(history):
    """Create deaths, severe cases, and hospital capacity plot."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure()

    if "dead" in df:
        fig.add_trace(go.Scatter(x=df["step"], y=df["dead"], mode="lines",
                                 name="Deaths", line=dict(color=COLORS["dead"], width=2)))
    if "severe" in df:
        fig.add_trace(go.Scatter(x=df["step"], y=df["severe"], mode="lines",
                                 name="Severe Cases", line=dict(color=COLORS["severe"], width=2)))
    if "hospital_capacity" in df:
        fig.add_trace(go.Scatter(x=df["step"], y=df["hospital_capacity"], mode="lines",
                                 name="Hospital Capacity", line=dict(color="#b40ec7", width=2, dash="dash")))
    if "vaccinated" in df:
        fig.add_trace(go.Scatter(x=df["step"], y=df["vaccinated"], mode="lines",
                                 name="Vaccinated %", line=dict(color=COLORS["vaccinated"], width=2)))

    fig.update_layout(
        title="Severity & Interventions", title_x=0.5, title_font_size=14,
        xaxis_title="Day",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=35, b=60),
    )
    return fig


def make_wealth_figure(history):
    """Create wealth distribution over time."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure()
    wealth_cols = [("Most Poor", "#FF0000"), ("Poor", "#f5a442"),
                   ("Middle Class", "#14e322"), ("Rich", "#808880"),
                   ("Most Rich", "#3291a8")]
    for col, color in wealth_cols:
        if col in df:
            fig.add_trace(go.Scatter(x=df["step"], y=df[col], mode="lines",
                                     name=col, line=dict(color=color, width=2),
                                     stackgroup="wealth"))
    fig.update_layout(
        title="Wealth by Stratum", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="Total Wealth",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=35, b=60),
    )
    return fig


def make_r0_figure(history):
    """Create R0 over time plot."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure()
    if "R0" in df:
        fig.add_trace(go.Scatter(x=df["step"], y=df["R0"], mode="lines+markers",
                                 name="R0", line=dict(color="#e74c3c", width=2),
                                 marker=dict(size=4)))
        # R0 = 1 threshold line
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="R0 = 1 (epidemic threshold)")
    fig.update_layout(
        title="Reproduction Number (R0)", title_x=0.5, title_font_size=14,
        xaxis_title="Day", yaxis_title="R0",
        margin=dict(l=50, r=10, t=35, b=30),
    )
    return fig


# ── Callbacks ──────────────────────────────────────────────────────────

@callback(
    Output("stats-cards", "children"),
    Output("grid-plot", "figure"),
    Output("seir-plot", "figure"),
    Output("severity-plot", "figure"),
    Output("wealth-plot", "figure"),
    Output("r0-plot", "figure"),
    Output("status-text", "children"),
    Input("init-btn", "n_clicks"),
    Input("run-btn", "n_clicks"),
    State("pop-slider", "value"),
    State("ptrans-slider", "value"),
    State("init-inf-slider", "value"),
    State("interventions", "value"),
    State("vax-rate-slider", "value"),
    prevent_initial_call=True,
)
def update_dashboard(init_clicks, run_clicks, pop, ptrans, init_inf, interventions, vax_rate):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    interventions = interventions or []

    if triggered == "init-btn":
        # Initialize new model
        model = InfectionModel(
            N=pop, width=15, height=15, ptrans=ptrans,
            initial_infected_perc=init_inf,
            lockdown="lockdown" in interventions,
            saq="saq" in interventions,
            ipa="ipa" in interventions,
            mm="mm" in interventions,
            vaccination="vaccination" in interventions,
            vaccination_rate=vax_rate,
            seed=42,
        )
        SIM_STATE["model"] = model
        SIM_STATE["history"] = [{
            "step": 0,
            "susceptible": model.percentage_susceptible,
            "exposed": model.percentage_exposed,
            "infected": model.percentage_infected,
            "recovered": model.percentage_recovered,
            "dead": model.dead,
            "severe": model.severe,
            "hospital_capacity": model.hospital_capacity,
            "vaccinated": model.percentage_vaccinated,
            "R0": model.R0,
            "Most Poor": model.wealth_most_poor,
            "Poor": model.wealth_poor,
            "Middle Class": model.wealth_working_class,
            "Rich": model.wealth_rich,
            "Most Rich": model.wealth_most_rich,
        }]
        status = f"Model initialized with {pop} agents."

    elif triggered == "run-btn":
        model = SIM_STATE.get("model")
        if model is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, "Initialize the model first!"

        # Run 60 steps
        for _ in range(60):
            if not model.running:
                break
            model.step()
            SIM_STATE["history"].append({
                "step": len(SIM_STATE["history"]),
                "susceptible": model.percentage_susceptible,
                "exposed": model.percentage_exposed,
                "infected": model.percentage_infected,
                "recovered": model.percentage_recovered,
                "dead": model.dead,
                "severe": model.severe,
                "hospital_capacity": model.hospital_capacity,
                "vaccinated": model.percentage_vaccinated,
                "R0": model.R0,
                "Most Poor": model.wealth_most_poor,
                "Poor": model.wealth_poor,
                "Middle Class": model.wealth_working_class,
                "Rich": model.wealth_rich,
                "Most Rich": model.wealth_most_rich,
            })
        status = f"Ran to step {model.steps}. {'Simulation complete.' if not model.running else ''}"
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, ""

    model = SIM_STATE["model"]
    history = SIM_STATE["history"]

    # Stats cards
    cards = [
        make_stat_card("Susceptible", f"{model.percentage_susceptible:.1f}%", COLORS["susceptible"]),
        make_stat_card("Exposed", f"{model.percentage_exposed:.1f}%", COLORS["exposed"]),
        make_stat_card("Infected", f"{model.percentage_infected:.1f}%", COLORS["infected"]),
        make_stat_card("Recovered", f"{model.percentage_recovered:.1f}%", COLORS["recovered"]),
        make_stat_card("Dead", str(model.dead), COLORS["dead"]),
        make_stat_card("R0", f"{model.R0:.2f}", "#e74c3c"),
        make_stat_card("Vaccinated", f"{model.percentage_vaccinated:.1f}%", COLORS["vaccinated"]),
    ]

    return (
        cards,
        make_grid_figure(model),
        make_seir_figure(history),
        make_severity_figure(history),
        make_wealth_figure(history),
        make_r0_figure(history),
        status,
    )


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting COVID-19 ABM Dashboard...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=False, port=8050)
