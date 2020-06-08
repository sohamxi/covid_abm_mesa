from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import TextElement
from model import InfectionModel, InfectionState

def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if agent.state == 1:
        portrayal["Color"] = "#008000"
        portrayal["Layer"] = 0
    elif agent.state == 2:
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.2
    else:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.2
    return portrayal


class MyTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        infected = model.infected
        r_o = model.R0
        recovered = model.recovered

        return "Number of Infected cases: {}<br>Number of Recovered cases: {}<br>R0 value: {}".format(
            infected, recovered, r_o
        )

canvas_element = CanvasGrid(agent_portrayal, 20, 20, 200, 200)
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

model_params = {
    "N": UserSettableParameter(
        "slider",
        "Number of agents",
        100,
        2,
        200,
        1,
        description="Choose how many agents to include in the model",
    ),
    "ptrans": UserSettableParameter("slider", "Transmission Probability", 0.1,0.2, 1.0, 0.1),
    "death_rate": UserSettableParameter("slider", "Death Rate", 0.01, 0.00, 1.0, 0.05)
}

server = ModularServer(
    InfectionModel, [canvas_element, text_element, chart], "Covid Model", model_params
)
server.port = 8521