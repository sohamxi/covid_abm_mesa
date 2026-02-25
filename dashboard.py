"""
COVID-19 ABM Dashboard

Three-page dashboard for exploring COVID-19 intervention strategies:
  1. About     — explains the model, charts, and key terms
  2. Simulate  — run a single scenario interactively
  3. Compare   — side-by-side scenario comparison

Launch:
    python dashboard.py
    Open http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from model import InfectionModel
from agent import InfectionState, InfectionSeverity

# ── App setup ──────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="COVID-19 ABM Dashboard",
                suppress_callback_exceptions=True)

# Each agent represents SCALE_FACTOR people. The slider shows real-world
# population sizes (50K–500K) but the model runs a representative sample.
SCALE_FACTOR = 1000

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

STATE_COLORS = {
    "susceptible": "#b5551d",
    "exposed": "#fcf00a",
    "infected": "#fa0c00",
    "recovered": "#008000",
    "dead": "#121010",
    "vaccinated": "#4a90d9",
    "severe": "#8b0000",
}

# ── Global state for single-simulation tab ────────────────────────────

SIM_STATE = {"model": None, "history": []}


# ── Helper: info icon with hover tooltip ──────────────────────────────

def info_icon(tooltip):
    """Small circled (i) that shows a tooltip on hover."""
    return html.Span(
        "i",
        title=tooltip,
        style={
            "display": "inline-flex", "alignItems": "center",
            "justifyContent": "center", "width": "16px", "height": "16px",
            "borderRadius": "50%", "backgroundColor": "#95a5a6",
            "color": "white", "fontSize": "10px", "fontStyle": "italic",
            "fontWeight": "bold", "marginLeft": "6px", "cursor": "help",
            "verticalAlign": "middle", "userSelect": "none",
        },
    )


def section_label(text, tooltip):
    """Label text followed by an info icon."""
    return html.Div([
        html.Span(text, style={"fontWeight": "600", "fontSize": "13px"}),
        info_icon(tooltip),
    ], style={"marginBottom": "4px", "marginTop": "10px"})


def stat_card(label, value, color="#2c3e50", tooltip=""):
    """Small stat card with optional tooltip."""
    children = [
        html.Div(str(value), style={"fontSize": "20px", "fontWeight": "bold", "color": color}),
        html.Div([
            html.Span(label, style={"fontSize": "11px", "opacity": "0.7"}),
            info_icon(tooltip) if tooltip else None,
        ]),
    ]
    return html.Div(children, style={
        "padding": "8px 14px", "backgroundColor": "#fff",
        "borderRadius": "6px", "border": f"2px solid {color}",
        "textAlign": "center", "minWidth": "80px",
    })


def pop_display(value):
    """Format population number for display: 200000 -> '200K'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value // 1_000}K"
    return str(value)


# ── Population slider (shared by Simulate and Compare tabs) ───────────

def pop_slider(id_prefix):
    return dcc.Slider(
        id=f"{id_prefix}-pop-slider",
        min=50_000, max=500_000, step=50_000, value=200_000,
        marks={50_000: "50K", 200_000: "200K", 350_000: "350K", 500_000: "500K"},
    )


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — ABOUT
# ══════════════════════════════════════════════════════════════════════

about_layout = html.Div([
    html.Div([
        html.Div([
            html.H2("What is this?", style={"marginTop": "0"}),
            html.P([
                "This is an ", html.Strong("Agent-Based Model (ABM)"),
                " that simulates COVID-19 spreading through a population. "
                "Each person in the simulation is an individual agent with their own age, "
                "job, household, and health status. The model tracks how the disease "
                "spreads through realistic contact networks (homes, workplaces, schools, "
                "and the community) and how different interventions change the outcome."
            ]),
            html.P(
                "The goal is to help compare intervention strategies — lockdowns, masks, "
                "vaccination, screening — and see their health AND economic tradeoffs "
                "before committing to a policy."
            ),

            html.Hr(),
            html.H3("How the model works"),
            html.Ul([
                html.Li([html.Strong("SEIR disease model: "),
                         "People move through Susceptible → Exposed → Infected → Recovered (or Died). "
                         "Each transition uses age-stratified probabilities from real COVID-19 data."]),
                html.Li([html.Strong("Contact networks: "),
                         "Agents interact in 4 layers — household (always active), workplace, school, "
                         "and community (grid-based). Each layer has a different transmission multiplier."]),
                html.Li([html.Strong("Vaccination: "),
                         "Prioritizes elderly, with dose-1 → dose-2 progression and waning efficacy."]),
                html.Li([html.Strong("Economic model: "),
                         "Agents earn and spend based on their social stratum (Lorenz curve). "
                         "Lockdowns and illness reduce economic activity."]),
            ]),

            html.Hr(),
            html.H3("How to read the charts"),
            html.Div([
                html.H4("SEIR Curves"),
                html.P(
                    "Shows the percentage of the population in each disease state over time. "
                    "A steep infected curve means a fast outbreak — interventions aim to "
                    "'flatten' this curve. When the recovered line rises, herd immunity is building."
                ),
                html.H4("Cumulative Deaths"),
                html.P(
                    "Total deaths over time. A curve that plateaus early means the outbreak "
                    "was contained. A curve that keeps climbing means ongoing, uncontrolled spread."
                ),
                html.H4("R0 — Reproduction Number"),
                html.P([
                    "The average number of people each infected person spreads the disease to. ",
                    html.Strong("R0 > 1"), " means the epidemic is growing. ",
                    html.Strong("R0 < 1"), " means it's shrinking. "
                    "The dashed line at R0 = 1 is the epidemic threshold — effective "
                    "interventions push R0 below this line."
                ]),
                html.H4("Wealth by Social Stratum"),
                html.P(
                    "Total accumulated wealth for each income group. Under lockdowns and "
                    "high infection, the poorest strata lose the most relative wealth. "
                    "This chart reveals the economic equity impact of each intervention."
                ),
                html.H4("Agent Grid"),
                html.P(
                    "A spatial view of agents on the simulation grid. Colors show disease "
                    "state: orange = susceptible, yellow = exposed, red = infected, "
                    "green = recovered, black = dead, blue = vaccinated, dark red = severe."
                ),
            ], style={"paddingLeft": "10px"}),

            html.Hr(),
            html.H3("Key terms"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Term", style={"textAlign": "left", "padding": "6px 12px"}),
                    html.Th("Meaning", style={"textAlign": "left", "padding": "6px 12px"}),
                ])),
                html.Tbody([
                    html.Tr([html.Td("Transmission Probability", style={"padding": "6px 12px"}),
                             html.Td("Base chance of infection per contact. Higher = faster spread.",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Incubation Period", style={"padding": "6px 12px"}),
                             html.Td("Days from exposure to becoming infectious (Exposed → Infected).",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Graduated Lockdown", style={"padding": "6px 12px"}),
                             html.Td("Mobility restrictions that tighten as infection rises: "
                                     ">10% infected → 90% reduction, >5% → 70%, >2% → 50%.",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Screening & Quarantine", style={"padding": "6px 12px"}),
                             html.Td("Reduces time to symptom detection to 3 days, "
                                     "triggering earlier quarantine and fewer onward infections.",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Public Awareness", style={"padding": "6px 12px"}),
                             html.Td("Reduces transmission by 30% (behavior change, hygiene).",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Mandatory Masks", style={"padding": "6px 12px"}),
                             html.Td("Reduces transmission by 50%. Stacks with public awareness.",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Hospital Capacity", style={"padding": "6px 12px"}),
                             html.Td("When severe cases exceed capacity, death rate triples.",
                                     style={"padding": "6px 12px"})]),
                    html.Tr([html.Td("Social Stratum", style={"padding": "6px 12px"}),
                             html.Td("Income group based on Lorenz curve wealth distribution "
                                     "(Most Poor → Most Rich).",
                                     style={"padding": "6px 12px"})]),
                ]),
            ], style={"borderCollapse": "collapse", "width": "100%",
                      "border": "1px solid #ddd"}),

            html.Hr(),
            html.H3("What to expect"),
            html.Ul([
                html.Li("With no intervention, expect a sharp outbreak peaking around day 15–30, "
                        "high deaths, and significant wealth loss for the poorest."),
                html.Li("Lockdown flattens the infection curve but causes severe economic damage."),
                html.Li("Masks + awareness are cost-effective — moderate health gains, minimal "
                        "economic disruption."),
                html.Li("Vaccination alone is slow to take effect but provides lasting protection."),
                html.Li("Full response (all interventions) gives the best health outcomes but "
                        "at the highest economic cost. The comparison dashboard helps you see "
                        "whether the tradeoff is worth it."),
            ]),
            html.P([
                html.Strong("Note: "),
                f"Each simulated agent represents ~{SCALE_FACTOR:,} people. Percentages and "
                "rates are accurate; absolute counts (deaths) are scaled proportionally. "
                "Results vary between runs due to stochastic (random) effects — use "
                "multiple runs in the Compare tab for reliable averages."
            ], style={"backgroundColor": "#fff3cd", "padding": "12px",
                      "borderRadius": "6px", "border": "1px solid #ffc107",
                      "marginTop": "15px"}),
        ], style={"maxWidth": "900px", "margin": "0 auto", "lineHeight": "1.7"}),
    ], style={"padding": "25px", "overflowY": "auto",
              "height": "calc(100vh - 120px)"}),
])


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — SINGLE SIMULATION
# ══════════════════════════════════════════════════════════════════════

sim_layout = html.Div([
    html.Div([
        # Left sidebar
        html.Div([
            html.H3("Parameters", style={"marginTop": "0"}),

            section_label("Population",
                          "Total population size. Each simulated agent represents "
                          f"~{SCALE_FACTOR:,} real people."),
            pop_slider("sim"),

            section_label("Transmission Probability",
                          "Base probability of infection per contact event. "
                          "COVID-19 estimates range from 0.03–0.10."),
            dcc.Slider(id="sim-ptrans-slider", min=0.01, max=0.2, step=0.01,
                       value=0.05, marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2"}),

            section_label("Initial Infected %",
                          "Fraction of the population infected at day 0. "
                          "Represents imported cases seeding the outbreak."),
            dcc.Slider(id="sim-init-inf-slider", min=0.01, max=0.2, step=0.01,
                       value=0.05, marks={0.01: "1%", 0.1: "10%", 0.2: "20%"}),

            section_label("Simulation Days",
                          "Number of days to simulate. Longer runs show long-term "
                          "effects like waning immunity and economic recovery."),
            dcc.Slider(id="sim-days-slider", min=30, max=180, step=10, value=60,
                       marks={30: "30", 60: "60", 120: "120", 180: "180"}),

            html.Hr(),
            html.H3("Interventions"),

            dcc.Checklist(id="sim-interventions", options=[
                {"label": " Lockdown (graduated)", "value": "lockdown"},
                {"label": " Screening & Quarantine", "value": "saq"},
                {"label": " Public Awareness (-30% trans.)", "value": "ipa"},
                {"label": " Mandatory Masks (-50% trans.)", "value": "mm"},
                {"label": " Vaccination", "value": "vaccination"},
            ], value=[], style={"lineHeight": "2.2"}),

            section_label("Vaccination Rate (daily)",
                          "Fraction of susceptible population vaccinated per day. "
                          "0.02 = 2% of unvaccinated people get a shot each day."),
            dcc.Slider(id="sim-vax-rate-slider", min=0.005, max=0.05, step=0.005,
                       value=0.02, marks={0.005: "0.5%", 0.02: "2%", 0.05: "5%"}),

            html.Hr(),
            html.Button("Run Simulation", id="sim-run-btn", n_clicks=0,
                        style={"width": "100%", "padding": "12px", "fontSize": "15px",
                               "backgroundColor": "#3498db", "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "bold"}),

            html.Div(id="sim-status",
                     style={"marginTop": "10px", "fontSize": "13px", "color": "#666"}),

        ], style={"width": "280px", "padding": "15px", "backgroundColor": "#fff",
                  "borderRight": "1px solid #ddd", "overflowY": "auto"}),

        # Right: charts
        html.Div([
            # Stat cards
            html.Div(id="sim-stats", style={
                "display": "flex", "gap": "8px", "marginBottom": "12px", "flexWrap": "wrap",
            }),

            # Grid + SEIR
            html.Div([
                html.Div([dcc.Graph(id="sim-grid-plot", style={"height": "340px"})],
                         style={"flex": "1", "minWidth": "380px"}),
                html.Div([dcc.Graph(id="sim-seir-plot", style={"height": "340px"})],
                         style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),

            # Severity + Wealth
            html.Div([
                html.Div([dcc.Graph(id="sim-severity-plot", style={"height": "300px"})],
                         style={"flex": "1", "minWidth": "380px"}),
                html.Div([dcc.Graph(id="sim-wealth-plot", style={"height": "300px"})],
                         style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                      "marginTop": "10px"}),

            # R0
            html.Div([dcc.Graph(id="sim-r0-plot", style={"height": "250px"})],
                     style={"marginTop": "10px"}),

        ], style={"flex": "1", "padding": "15px", "overflowY": "auto"}),
    ], style={"display": "flex", "height": "calc(100vh - 120px)"}),
])


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════

compare_layout = html.Div([
    html.Div([
        # Left sidebar
        html.Div([
            html.H3("Shared Parameters", style={"marginTop": "0"}),

            section_label("Population",
                          f"Total population. Each agent represents ~{SCALE_FACTOR:,} people."),
            pop_slider("cmp"),

            section_label("Transmission Probability",
                          "Base infection probability per contact, shared across all scenarios."),
            dcc.Slider(id="cmp-ptrans-slider", min=0.01, max=0.2, step=0.01,
                       value=0.05, marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2"}),

            section_label("Initial Infected %",
                          "Starting outbreak size, same for all scenarios."),
            dcc.Slider(id="cmp-init-inf-slider", min=0.01, max=0.2, step=0.01,
                       value=0.05, marks={0.01: "1%", 0.1: "10%", 0.2: "20%"}),

            section_label("Simulation Days",
                          "Duration of each scenario run."),
            dcc.Slider(id="cmp-days-slider", min=30, max=180, step=10, value=60,
                       marks={30: "30", 60: "60", 120: "120", 180: "180"}),

            section_label("Runs per Scenario",
                          "Number of times to repeat each scenario with different "
                          "random seeds. Results are averaged to reduce noise."),
            dcc.Slider(id="cmp-runs-slider", min=1, max=5, step=1, value=3,
                       marks={1: "1", 3: "3", 5: "5"}),

            html.Hr(),
            html.Div([
                html.H3("Scenarios to Compare", style={"display": "inline"}),
                info_icon("Check the scenarios you want to run. All will use the same "
                          "shared parameters above, so differences are purely from the "
                          "intervention strategy."),
            ]),

            dcc.Checklist(
                id="cmp-scenario-checklist",
                options=[{"label": f"  {name}", "value": name} for name in SCENARIOS],
                value=list(SCENARIOS.keys()),
                style={"lineHeight": "2.2"},
            ),

            html.Hr(),
            html.Button("Run Comparison", id="cmp-run-btn", n_clicks=0,
                        style={"width": "100%", "padding": "12px", "fontSize": "15px",
                               "backgroundColor": "#27ae60", "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "bold"}),

            html.Div(id="cmp-status",
                     style={"marginTop": "10px", "fontSize": "13px", "color": "#666"}),

        ], style={"width": "280px", "padding": "15px", "backgroundColor": "#fff",
                  "borderRight": "1px solid #ddd", "overflowY": "auto"}),

        # Right: comparison charts
        html.Div([
            html.Div(id="cmp-summary-table", style={"marginBottom": "15px"}),

            html.Div([
                html.Div([dcc.Graph(id="cmp-infection-plot", style={"height": "320px"})],
                         style={"flex": "1", "minWidth": "400px"}),
                html.Div([dcc.Graph(id="cmp-deaths-plot", style={"height": "320px"})],
                         style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([dcc.Graph(id="cmp-r0-plot", style={"height": "320px"})],
                         style={"flex": "1", "minWidth": "400px"}),
                html.Div([dcc.Graph(id="cmp-wealth-plot", style={"height": "320px"})],
                         style={"flex": "1", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                      "marginTop": "10px"}),

        ], style={"flex": "1", "padding": "15px", "overflowY": "auto"}),
    ], style={"display": "flex", "height": "calc(100vh - 120px)"}),
])


# ══════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — TABS
# ══════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COVID-19 Agent-Based Model",
                style={"margin": "0", "fontSize": "22px"}),
        html.P("Simulate, compare, and understand intervention strategies",
               style={"margin": "0", "fontSize": "12px", "opacity": "0.7"}),
    ], style={"padding": "12px 20px", "backgroundColor": "#2c3e50",
              "color": "white"}),

    # Tabs
    dcc.Tabs(id="main-tabs", value="about", children=[
        dcc.Tab(label="About", value="about",
                style={"padding": "8px 20px"}, selected_style={"padding": "8px 20px", "fontWeight": "bold"}),
        dcc.Tab(label="Simulate", value="simulate",
                style={"padding": "8px 20px"}, selected_style={"padding": "8px 20px", "fontWeight": "bold"}),
        dcc.Tab(label="Compare Scenarios", value="compare",
                style={"padding": "8px 20px"}, selected_style={"padding": "8px 20px", "fontWeight": "bold"}),
    ], style={"borderBottom": "2px solid #2c3e50"}),

    # Tab content
    html.Div(id="tab-content"),

], style={"fontFamily": "system-ui, -apple-system, sans-serif",
          "backgroundColor": "#f8f9fa", "height": "100vh", "overflow": "hidden"})


# ── Tab routing ────────────────────────────────────────────────────────

@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "about":
        return about_layout
    elif tab == "simulate":
        return sim_layout
    elif tab == "compare":
        return compare_layout
    return about_layout


# ══════════════════════════════════════════════════════════════════════
#  CALLBACKS — SINGLE SIMULATION
# ══════════════════════════════════════════════════════════════════════

def _collect_step(model):
    """Snapshot one step of model state into a dict."""
    return {
        "step": model.steps,
        "susceptible": model.percentage_susceptible,
        "exposed": model.percentage_exposed,
        "infected": model.percentage_infected,
        "recovered": model.percentage_recovered,
        "dead": model.dead * SCALE_FACTOR,
        "severe": model.severe * SCALE_FACTOR,
        "hospital_capacity": model.hospital_capacity * SCALE_FACTOR,
        "vaccinated": model.percentage_vaccinated,
        "R0": model.R0,
        "Most Poor": model.wealth_most_poor,
        "Poor": model.wealth_poor,
        "Middle Class": model.wealth_working_class,
        "Rich": model.wealth_rich,
        "Most Rich": model.wealth_most_rich,
    }


def _make_grid_figure(model):
    """Scatter plot of agents colored by disease state."""
    if model is None:
        return go.Figure()
    x, y, colors, sizes, text = [], [], [], [], []
    cmap = {
        InfectionState.SUSCEPTIBLE: STATE_COLORS["susceptible"],
        InfectionState.EXPOSED: STATE_COLORS["exposed"],
        InfectionState.INFECTED: STATE_COLORS["infected"],
        InfectionState.RECOVERED: STATE_COLORS["recovered"],
        InfectionState.DIED: STATE_COLORS["dead"],
    }
    for a in model.agents:
        if a.pos is None:
            continue
        x.append(a.pos[0])
        y.append(a.pos[1])
        c = cmap.get(a.state, "#999")
        if a.vaccinated and a.state == InfectionState.SUSCEPTIBLE:
            c = STATE_COLORS["vaccinated"]
        if a.severity == InfectionSeverity.Severe:
            c = STATE_COLORS["severe"]
        colors.append(c)
        sizes.append(12 if a.severity == InfectionSeverity.Severe else 7)
        text.append(f"Age: {a.age:.0f} | {a.state.name}")
    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors, size=sizes, line=dict(width=0.5, color="white")),
        text=text, hoverinfo="text",
    ))
    fig.update_layout(
        title={"text": "Agent Grid  \u24d8", "x": 0.5, "font": {"size": 14}},
        xaxis=dict(range=[-0.5, model.grid.width - 0.5], showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.5, model.grid.height - 0.5], showgrid=False, zeroline=False),
        plot_bgcolor="#f0f0f0", margin=dict(l=30, r=10, t=35, b=30),
    )
    return fig


@callback(
    Output("sim-stats", "children"),
    Output("sim-grid-plot", "figure"),
    Output("sim-seir-plot", "figure"),
    Output("sim-severity-plot", "figure"),
    Output("sim-wealth-plot", "figure"),
    Output("sim-r0-plot", "figure"),
    Output("sim-status", "children"),
    Input("sim-run-btn", "n_clicks"),
    State("sim-pop-slider", "value"),
    State("sim-ptrans-slider", "value"),
    State("sim-init-inf-slider", "value"),
    State("sim-days-slider", "value"),
    State("sim-interventions", "value"),
    State("sim-vax-rate-slider", "value"),
    prevent_initial_call=True,
)
def run_single_sim(n_clicks, pop_display_val, ptrans, init_inf, max_steps,
                   interventions, vax_rate):
    interventions = interventions or []
    n_agents = pop_display_val // SCALE_FACTOR

    model = InfectionModel(
        N=n_agents, width=15, height=15, ptrans=ptrans,
        initial_infected_perc=init_inf, max_steps=max_steps,
        lockdown="lockdown" in interventions,
        saq="saq" in interventions,
        ipa="ipa" in interventions,
        mm="mm" in interventions,
        vaccination="vaccination" in interventions,
        vaccination_rate=vax_rate,
        seed=42,
    )

    history = [_collect_step(model)]
    for _ in range(max_steps):
        if not model.running:
            break
        model.step()
        history.append(_collect_step(model))

    SIM_STATE["model"] = model
    SIM_STATE["history"] = history
    df = pd.DataFrame(history)

    # ── Stat cards ──
    scaled_dead = int(model.dead * SCALE_FACTOR)
    cards = [
        stat_card("Susceptible", f"{model.percentage_susceptible:.1f}%",
                  STATE_COLORS["susceptible"],
                  "People who haven't been infected yet and have no immunity."),
        stat_card("Exposed", f"{model.percentage_exposed:.1f}%",
                  STATE_COLORS["exposed"],
                  "Infected but not yet infectious. In incubation period."),
        stat_card("Infected", f"{model.percentage_infected:.1f}%",
                  STATE_COLORS["infected"],
                  "Currently infectious and able to spread the disease."),
        stat_card("Recovered", f"{model.percentage_recovered:.1f}%",
                  STATE_COLORS["recovered"],
                  "Recovered with temporary natural immunity (wanes over time)."),
        stat_card("Deaths", f"{scaled_dead:,}",
                  STATE_COLORS["dead"],
                  f"Total deaths (scaled — each agent = ~{SCALE_FACTOR:,} people)."),
        stat_card("R0", f"{model.R0:.2f}", "#e74c3c",
                  "Reproduction number. >1 = epidemic growing, <1 = shrinking."),
        stat_card("Vaccinated", f"{model.percentage_vaccinated:.1f}%",
                  STATE_COLORS["vaccinated"],
                  "Percentage of population that has received at least one dose."),
    ]

    # ── SEIR ──
    seir_fig = go.Figure()
    for col, color, name in [
        ("susceptible", STATE_COLORS["susceptible"], "Susceptible"),
        ("exposed", STATE_COLORS["exposed"], "Exposed"),
        ("infected", STATE_COLORS["infected"], "Infected"),
        ("recovered", STATE_COLORS["recovered"], "Recovered"),
    ]:
        seir_fig.add_trace(go.Scatter(
            x=df["step"], y=df[col], mode="lines",
            name=name, line=dict(color=color, width=2),
        ))
    seir_fig.update_layout(
        title={"text": "SEIR Curves (% of population)", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="% Population",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=55),
    )

    # ── Severity / Deaths ──
    sev_fig = go.Figure()
    sev_fig.add_trace(go.Scatter(x=df["step"], y=df["dead"], mode="lines",
                                 name="Deaths", line=dict(color=STATE_COLORS["dead"], width=2)))
    sev_fig.add_trace(go.Scatter(x=df["step"], y=df["severe"], mode="lines",
                                 name="Severe Cases", line=dict(color=STATE_COLORS["severe"], width=2)))
    sev_fig.add_trace(go.Scatter(x=df["step"], y=df["hospital_capacity"], mode="lines",
                                 name="Hospital Capacity",
                                 line=dict(color="#b40ec7", width=2, dash="dash")))
    sev_fig.add_trace(go.Scatter(x=df["step"], y=df["vaccinated"], mode="lines",
                                 name="Vaccinated %",
                                 line=dict(color=STATE_COLORS["vaccinated"], width=2)))
    sev_fig.update_layout(
        title={"text": "Severity & Interventions", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── Wealth ──
    wealth_fig = go.Figure()
    for col, color in [("Most Poor", "#FF0000"), ("Poor", "#f5a442"),
                       ("Middle Class", "#14e322"), ("Rich", "#808880"),
                       ("Most Rich", "#3291a8")]:
        wealth_fig.add_trace(go.Scatter(
            x=df["step"], y=df[col], mode="lines",
            name=col, line=dict(color=color, width=2), stackgroup="wealth",
        ))
    wealth_fig.update_layout(
        title={"text": "Wealth by Social Stratum", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="Total Wealth",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── R0 ──
    r0_fig = go.Figure()
    r0_fig.add_trace(go.Scatter(x=df["step"], y=df["R0"], mode="lines+markers",
                                name="R0", line=dict(color="#e74c3c", width=2),
                                marker=dict(size=3)))
    r0_fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="R0 = 1 (epidemic threshold)")
    r0_fig.update_layout(
        title={"text": "Reproduction Number (R0)", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="R0",
        margin=dict(l=50, r=10, t=40, b=30),
    )

    # Grid
    grid_fig = _make_grid_figure(model)

    pop_label = pop_display(pop_display_val)
    status = (f"Simulated {pop_label} population ({n_agents} agents) "
              f"for {model.steps} days. "
              f"{'Simulation complete.' if not model.running else ''}")

    return cards, grid_fig, seir_fig, sev_fig, wealth_fig, r0_fig, status


# ══════════════════════════════════════════════════════════════════════
#  CALLBACKS — SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════

def _run_scenario(name, scenario_params, shared_params, n_runs, max_steps):
    """Run one scenario n_runs times and return averaged time-series."""
    all_runs = []
    n_agents = shared_params["N"]
    for run_idx in range(n_runs):
        params = {
            "N": n_agents, "width": 15, "height": 15,
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

    combined = pd.concat(all_runs)
    averaged = combined.groupby("step").mean(numeric_only=True).reset_index()
    averaged["scenario"] = name
    return averaged


@callback(
    Output("cmp-summary-table", "children"),
    Output("cmp-infection-plot", "figure"),
    Output("cmp-deaths-plot", "figure"),
    Output("cmp-r0-plot", "figure"),
    Output("cmp-wealth-plot", "figure"),
    Output("cmp-status", "children"),
    Input("cmp-run-btn", "n_clicks"),
    State("cmp-pop-slider", "value"),
    State("cmp-ptrans-slider", "value"),
    State("cmp-init-inf-slider", "value"),
    State("cmp-days-slider", "value"),
    State("cmp-runs-slider", "value"),
    State("cmp-scenario-checklist", "value"),
    prevent_initial_call=True,
)
def run_comparison(n_clicks, pop_display_val, ptrans, init_inf, max_steps,
                   n_runs, selected_scenarios):
    if not selected_scenarios:
        return (no_update, no_update, no_update, no_update, no_update,
                "Select at least one scenario.")

    n_agents = pop_display_val // SCALE_FACTOR
    shared = {"N": n_agents, "ptrans": ptrans, "initial_infected_perc": init_inf}

    all_results = []
    for name in selected_scenarios:
        result = _run_scenario(name, SCENARIOS.get(name, {}), shared, n_runs, max_steps)
        all_results.append(result)
    results_df = pd.concat(all_results, ignore_index=True)

    # ── Summary table ──
    rows = []
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        final = sc.iloc[-1]
        peak_inf = sc["infected"].max()
        peak_day = int(sc.loc[sc["infected"].idxmax(), "step"])
        scaled_dead = final["dead"] * SCALE_FACTOR
        rows.append({
            "Scenario": name,
            "Peak Infected (%)": f"{peak_inf:.1f}",
            "Peak Day": str(peak_day),
            f"Total Deaths (x{SCALE_FACTOR})": f"{scaled_dead:,.0f}",
            "Peak R0": f"{sc['R0'].max():.2f}",
            "Final Vaccinated (%)": f"{final.get('vaccinated', 0):.1f}",
            "Wealth (Most Poor)": f"{final.get('Most Poor', 0):,.0f}",
            "Wealth (Most Rich)": f"{final.get('Most Rich', 0):,.0f}",
        })

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in rows[0]],
        data=rows,
        style_header={"backgroundColor": "#2c3e50", "color": "white",
                      "fontWeight": "bold", "fontSize": "12px", "padding": "8px"},
        style_cell={"textAlign": "center", "padding": "8px", "fontSize": "12px",
                    "border": "1px solid #ddd"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )

    # ── Infection curves ──
    inf_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        inf_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["infected"], mode="lines",
            name=name, line=dict(color=SCENARIO_COLORS.get(name, "#999"), width=2.5),
        ))
    inf_fig.update_layout(
        title={"text": "Infected Population (%) Over Time", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="% Infected",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── Deaths ──
    death_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        death_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["dead"] * SCALE_FACTOR, mode="lines",
            name=name, line=dict(color=SCENARIO_COLORS.get(name, "#999"), width=2.5),
        ))
    death_fig.update_layout(
        title={"text": f"Cumulative Deaths (scaled x{SCALE_FACTOR})", "x": 0.5,
               "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="Deaths",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── R0 ──
    r0_fig = go.Figure()
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        r0_fig.add_trace(go.Scatter(
            x=sc["step"], y=sc["R0"], mode="lines",
            name=name, line=dict(color=SCENARIO_COLORS.get(name, "#999"), width=2.5),
        ))
    r0_fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="R0 = 1 (epidemic threshold)")
    r0_fig.update_layout(
        title={"text": "Reproduction Number (R0) Over Time", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Day", yaxis_title="R0",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=10, t=40, b=60),
    )

    # ── Wealth (grouped bar) ──
    wealth_fig = go.Figure()
    strata = ["Most Poor", "Poor", "Middle Class", "Rich", "Most Rich"]
    for name in selected_scenarios:
        sc = results_df[results_df["scenario"] == name]
        final = sc.iloc[-1]
        wealth_fig.add_trace(go.Bar(
            x=strata, y=[final.get(s, 0) for s in strata],
            name=name, marker_color=SCENARIO_COLORS.get(name, "#999"),
        ))
    wealth_fig.update_layout(
        title={"text": "Final Wealth by Social Stratum", "x": 0.5, "font": {"size": 14}},
        xaxis_title="Social Stratum", yaxis_title="Total Wealth",
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=50, r=10, t=40, b=70),
    )

    pop_label = pop_display(pop_display_val)
    total_runs = len(selected_scenarios) * n_runs
    status = (f"Ran {len(selected_scenarios)} scenarios x {n_runs} runs = "
              f"{total_runs} simulations ({pop_label} population, {max_steps} days each).")

    return table, inf_fig, death_fig, r0_fig, wealth_fig, status


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting COVID-19 ABM Dashboard...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=False, port=8050)
