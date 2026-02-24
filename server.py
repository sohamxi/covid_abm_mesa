from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model import InfectionModel
from agent import InfectionState, InfectionSeverity


def agent_portrayal(agent):
    portrayal = {"size": 25}

    if agent.state == InfectionState.SUSCEPTIBLE:
        if agent.vaccinated:
            portrayal["color"] = "#4a90d9"  # Blue: Vaccinated+Susceptible
        else:
            portrayal["color"] = "#b5551d"  # Orange: Susceptible
    elif agent.state == InfectionState.EXPOSED:
        portrayal["color"] = "#fcf00a"  # Yellow: Exposed
    elif agent.state == InfectionState.INFECTED:
        if agent.severity == InfectionSeverity.Severe:
            portrayal["color"] = "#8b0000"  # Dark Red: Severe
            portrayal["size"] = 35
        else:
            portrayal["color"] = "#fa0c00"  # Red: Infected
    elif agent.state == InfectionState.RECOVERED:
        portrayal["color"] = "#008000"  # Green: Recovered
    else:  # DIED
        portrayal["color"] = "#121010"  # Black: Dead
        portrayal["size"] = 10

    return portrayal


model_params = {
    "N": {
        "type": "SliderInt",
        "value": 200,
        "label": "Number of agents",
        "min": 20,
        "max": 1000,
        "step": 10,
    },
    "width": 15,
    "height": 15,
    "ptrans": {
        "type": "SliderFloat",
        "value": 0.05,
        "label": "Transmission Probability",
        "min": 0.01,
        "max": 0.5,
        "step": 0.01,
    },
    "death_rate": {
        "type": "SliderFloat",
        "value": 0.0193,
        "label": "Base Death Rate",
        "min": 0.001,
        "max": 0.1,
        "step": 0.001,
    },
    "initial_infected_perc": {
        "type": "SliderFloat",
        "value": 0.05,
        "label": "Initial Infected %",
        "min": 0.01,
        "max": 0.3,
        "step": 0.01,
    },
    "lockdown": {
        "type": "Checkbox",
        "value": False,
        "label": "Lockdown (graduated)",
    },
    "saq": {
        "type": "Checkbox",
        "value": False,
        "label": "Screening & Quarantine",
    },
    "ipa": {
        "type": "Checkbox",
        "value": False,
        "label": "Public Awareness (-30% transmission)",
    },
    "mm": {
        "type": "Checkbox",
        "value": False,
        "label": "Mandatory Masks (-50% transmission)",
    },
    "vaccination": {
        "type": "Checkbox",
        "value": False,
        "label": "Vaccination (elderly first)",
    },
    "vaccination_rate": {
        "type": "SliderFloat",
        "value": 0.02,
        "label": "Daily Vaccination Rate",
        "min": 0.005,
        "max": 0.1,
        "step": 0.005,
    },
}

page = SolaraViz(
    InfectionModel,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component(
            ["susceptible", "exposed", "infected", "recovered"]
        ),
        make_plot_component(
            ["dead", "severe_cases", "hospital", "vaccinated"]
        ),
        make_plot_component(
            ["Most Poor", "Poor", "Middle Class", "Rich", "Most Rich"]
        ),
    ],
    model_params=model_params,
    name="COVID-19 ABM",
)
