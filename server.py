from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import TextElement
from model import InfectionModel, InfectionState

def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if agent.state == 0: 
        portrayal["Color"] = "#b5551d" #Orange: Susceptible
        portrayal["Layer"] = 0
        portrayal["r"] = 0.9
    elif agent.state == 4:
        portrayal["Color"] = "#fcf00a" #Yellow: Exposed 
        portrayal["Layer"] = 1
        portrayal["r"] = 0.7
    elif agent.state == 1:
        portrayal["Color"] = "#fa0c00" #Red: Infected 
        portrayal["Layer"] = 2
        portrayal["r"] = 0.5
    elif agent.state == 2:
        portrayal["Color"] = "#008000" # Green:Recovered 
        portrayal["Layer"] = 3
        portrayal["r"] = 0.3
    else:
        portrayal["Color"] = '#121010' #Black: Dead
        portrayal["Layer"] = 4
        portrayal["r"] = 0.1
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
        exposed = model.exposed

        return "Number Suscpetible of  cases: {}<br>Number Exposed of Citizen: {}<br>Number of Infected cases: {}<br>Number of Recovered cases: {}<br>Dead: {}<br>R0 value: {}".format(
            susceptible,exposed, infected, recovered,dead, r_o
        )

canvas_element = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
text_element = MyTextElement()
chart = ChartModule(
    [
        {"Label": "susceptible", "Color": "#b5551d"},
        {"Label": "exposed", "Color": "#fcf00a"},
        {"Label": "infected", "Color": "#fa0c00"},
        {"Label": "recovered", "Color": "#008000"}
    ], data_collector_name="datacollector"
)

chart3 = ChartModule(
    [
        {"Label": "Most Poor", "Color": "#FF0000"},
        {"Label": "Poor", "Color": "#f5a442"},
        {"Label": "Middle Class", "Color": "#14e322"},
        {"Label": "Rich", "Color": "#808880"},
        {"Label": "Most Rich", "Color": '#3291a8'}
    ], data_collector_name="datacollector"
)

chart2 = ChartModule(
    [
        {"Label": "dead", "Color": "#121010"},
        {"Label": "severe_cases", "Color": '#99928e'},
        {"Label": "hospital", "Color": '#b40ec7'}
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
    InfectionModel, [canvas_element, text_element, chart, chart2, chart3], "Covid Model", model_params
)
server.port = 8521