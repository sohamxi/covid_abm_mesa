from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import TextElement
from model import InfectionModel, InfectionState

def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if agent.state == 1: 
        portrayal["Color"] = "#fa0c00" # Red:Infected
        portrayal["Layer"] = 0
        portrayal["r"] = 0.3
    elif agent.state == 0:
        portrayal["Color"] = "#008000" #Green: Susceptible 
        portrayal["Layer"] = 0
        portrayal["r"] = 0.5
    elif agent.state == 2:
        portrayal["Color"] = "grey" # Grey:Recovered 
        portrayal["Layer"] = 0
        portrayal["r"] = 0.5
    else:
        portrayal["Color"] = '#121010' #Black: Dead
        portrayal["Layer"] = 0
        portrayal["r"] = 0.5
    return portrayal


class MyTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        infected = model.infected
        r_o = model.R0
        recovered = model.recovered
        dead = model.dead
        susceptible = model.susceptible

        return "Number Suscpetible of  cases: {}<br>Number of Infected cases: {}<br>Number of Recovered cases: {}<br>Dead: {}<br>R0 value: {}".format(
            susceptible,infected, recovered,dead, r_o
        )

canvas_element = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
text_element = MyTextElement()
chart = ChartModule(
    [
        {"Label": "infected", "Color": "#FF0000"},
        {"Label": "susceptible", "Color": "#f5a442"},
        {"Label": "recovered", "Color": "#14e322"},
        {"Label": "dead", "Color": "#808880"},
        {"Label": "severe_cases", "Color": '#3291a8'}
    ], data_collector_name="datacollector"
)

chart2 = ChartModule(
    [
        {"Label": "Most Poor", "Color": "#FF0000"},
        {"Label": "Poor", "Color": "#f5a442"},
        {"Label": "Middle Class", "Color": "#14e322"},
        {"Label": "Rich", "Color": "#808880"},
        {"Label": "Most Rich", "Color": '#3291a8'}
    ], data_collector_name="datacollector"
)


model_params = {
    "N": UserSettableParameter(
        "slider",
        "Number of agents",
        100,
        2,
        500,
        1,
        description="Choose how many agents to include in the model",
    ),
    "width" : 10,
    "height" : 10,
    "ptrans": UserSettableParameter("slider", "Transmission Probability", 0.1,0.2, 1.0, 0.1),
    "death_rate": UserSettableParameter("slider", "Death Rate", 0.0193, 0.005, 0.4, 0.001),
    "lockdown" : UserSettableParameter("checkbox", "Lockdown", False),
    "saq" : UserSettableParameter("checkbox", "Screening and Quarantine", False),
    "ipa" : UserSettableParameter("checkbox","Increase Public Awareness", False),
    "mm" : UserSettableParameter("checkbox","Mandatory Masks", False)
}

server = ModularServer(
    InfectionModel, [canvas_element, text_element, chart, chart2], "Covid Model", model_params
)
server.port = 8521