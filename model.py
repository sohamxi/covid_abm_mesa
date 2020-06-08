import time,enum
from enum import Enum
import numpy as np
import pandas as pd
import pylab as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class InfectionState(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DIED = 3
    EXPOSED = 4
    

class InfectionSeverity(Enum):
    """
    The Severity of the Infected agents
    """
    Asymptomatic = 'a'
    Exposed = 'e'
    #Quarantined= 'q'
    Hospitalization = 'h'
    Severe = 's'

class JobType(Enum):
    GOVERNMENT = 'g'
    BLUE_COLLAR ='l'
    WHITE_COLLAR = 'e'
    UNEMPLOYED = 'u'
    BUSINESS_OWNER = 'b'

class SocialStratum(enum.IntEnum):
    """Dividing the Population into 5 quintiles """

    Most_Poor = 0
    Poor = 1
    Working_class = 2
    Rich = 3
    Most_Rich = 4

"""
Wealth distribution - Lorenz Curve
By quintile, source: https://www.worldbank.org/en/topic/poverty/lac-equity-lab1/income-inequality/composition-by-quintile
"""

lorenz_curve = [.04, .08, .13, .2, .55] ## wealth Distribution Based on Percentile (South American Nations)
share = np.min(lorenz_curve)
basic_income = np.array(lorenz_curve) / share

# TODO - Need to figure out how to restrict mobility (Lock down, Quarantine)

class Human(Agent):

    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # TODO - Age distribution to be taken as per location/ country
        #print(f'Inside Human')
        self.age = self.random.normalvariate(20,40)
        #print(f'Age Set')       
        self.state = InfectionState.SUSCEPTIBLE
        self.severity = InfectionSeverity.Exposed
        # TODO - Job type to affect Income and thus wealth
        #self.jobtype = JobType.WHITE_COLLAR
        #self.jobtype = np.random.choice([JobType.GOVERNMENT,JobType.BLUE_COLLAR,JobType.WHITE_COLLARJobType.UNEMPLOYED,JobType.BUSINESS_OWNER]) #p=[0.2,0.2,0.2,0.2,0.2] (Optional Based on Demography)
        self.infection_time = 0
        self.induced_infections = 0
        self.infected_others = False

        # Economic params
        self.social_stratum = self.random.choice([0,1,2,3,4])
        self.wealth = 0
        self.income = basic_income[self.social_stratum]
        self.expanditure = 0

    def getAgentIncome(self):
        """Calculate Agent's Income for the Step"""

        if (self.state != InfectionState.DIED) and (self.severity != InfectionSeverity.Severe) and (self.severity != InfectionSeverity.Hospitalization) :
            basic_income_temp = basic_income[self.social_stratum]
            variable_income = self.random.random() *self.random.random()* basic_income[self.social_stratum]
        else:
            basic_income_temp = 0
            variable_income = 0

        step_income = basic_income_temp + variable_income
        return step_income

    def getAgentExpense(self):
        """Calculate Agent's Expanditure for the step"""

        expense_temp = self.random.random() * basic_income[self.social_stratum]
        return expense_temp

    def update_Wealth(self):
        """Update Wealth of Agent in Current Step """

        self.income = self.getAgentIncome()
        self.expanditure = self.getAgentExpense()
        self.wealth = self.wealth + self.income - self.expanditure


    def move(self):
        """Move the agent"""

        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    
    
    
    def status(self):
        """Check infection status"""

        if self.state == InfectionState.INFECTED:     
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate,1-drate])
            # Check wheather Alive, if Dead remove from scheduler
            if alive == 0:
                self.state = InfectionState.DIED
                self.model.schedule.remove(self)
                self.model.dead_agents.append(self.unique_id)

            time_passed = self.model.schedule.time-self.infection_time
            if time_passed >= self.recovery_time:          
                self.state = InfectionState.RECOVERED

    
    
    
    def interact(self):
        """Interaction with other Agents"""

        neighbors = self.model.grid.get_cell_list_contents([self.pos])

        if len(neighbors)==0:
           print (self.unique_id + " is lonely")
        else :
          for other in neighbors:
            if other.state == InfectionState.DIED:
              pass
            else:
              self.infect(other)




    def infect(self,other):
        """Infect/Reinfect Other Agent"""

        if self.state == InfectionState.INFECTED :
          # check other Agent is Susceptible or Recovered : Susceptible
          if other.state == InfectionState.SUSCEPTIBLE:
            if self.random.random() < self.model.ptrans:
              other.state = InfectionState.INFECTED
              other.infection_time = self.model.schedule.time
              other.recovery_time = self.model.get_recovery_time()
              print(f'New Person Infected, recovery rate : {other.recovery_time}')
              self.induced_infections +=1
              self.infected_others = True
              # set Severity
              if self.random.random() < self.model.severe_perc:
                other.severity = InfectionSeverity.Severe
              else :
                other.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])


          # Reinfection Scenario
          elif other.state == InfectionState.RECOVERED:

            if self.random.random() < self.model.reinfection_rate:
              other.state = InfectionState.INFECTED
              other.infection_time = self.model.schedule.time
              other.recovery_time = self.model.get_recovery_time()
              self.induced_infections +=1
              self.infected_others = True
              # set Severity
              if self.random.random() < self.model.severe_perc:
                other.severity = InfectionSeverity.Severe
              else :
                other.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])


    def step(self):
        self.status()
        self.move()
        self.interact()
        self.update_Wealth()


class InfectionModel(Model):

    def __init__(self, N=10, width=10, height=10, ptrans = 0.25, reinfection_rate = 0.00,  severe_perc =0.18,
                 progression_period = 3, progression_sd = 2, death_rate = 0.0193, recovery_days = 21,
                 recovery_sd = 7, initial_infected_perc=0.2, initial_immune_perc = 0.01):
        self.population = N
        #self.model = model
        self.ptrans = ptrans
        self.reinfection_rate = reinfection_rate
        
        self.progression_period = progression_period
        self.progression_sd = progression_sd
        self.death_rate = death_rate
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.dead_agents = []

        self.severe_perc = severe_perc
        self.initial_infected_perc = initial_infected_perc
        self.initial_immune_perc = initial_immune_perc
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        

        self.susceptible = self.population
        self.dead = 0
        self.recovered = 0
        self.infected = 0
        self.R0 = 0
        self.severe = 0

        # Economic params Model related
        self.total_wealth = 10**4
        self.wealth_most_poor = lorenz_curve[0] * self.total_wealth
        self.wealth_poor = lorenz_curve[1] * self.total_wealth
        self.wealth_working_class = lorenz_curve[2] * self.total_wealth
        self.wealth_rich = lorenz_curve[3] * self.total_wealth
        self.wealth_most_rich = lorenz_curve[4] * self.total_wealth

        # Create Data Collecter for Aggregate Values  
        self.datacollector = DataCollector(model_reporters={"infected": 'infected',
                                                            "recovered": 'recovered',
                                                            "susceptible": 'susceptible',
                                                            "dead": 'dead',
                                                            "R0": 'R0',
                                                            "severe_cases": 'severe',
                                                            "Most Poor": 'wealth_most_poor',
                                                            "Poor": 'wealth_poor',
                                                            "Middle Class": 'wealth_working_class',
                                                            "Rich": 'wealth_rich',
                                                            "Most Rich": 'wealth_most_rich'})

        # Create Data Collecter for Aggregate Wealth Values  

        # Create Agents
        for i in range(self.population):
            a = Human(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            #print(f'Agent Added')
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            #print(f'Agent Palced')

            #Initial Infection (Make some Agents infected at start)
            infection = np.random.choice([0,1], p=[(1-self.initial_infected_perc), self.initial_infected_perc])
            if infection == 1:
                a.state = InfectionState.INFECTED
                a.recovery_time = self.get_recovery_time()
                #Severity Set
                severity = np.random.choice([0,1], p=[(1-self.severe_perc), self.severe_perc])
                if severity == 1:
                  a.severity = InfectionSeverity.Severe
                else :
                  a.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])
            #print(f'Agent Set')

        # Wealth Distributiom
        # Share the common wealth of 10^4 among the population, according each agent social stratum
        for quintile in [0, 1, 2, 3, 4]:
            total = lorenz_curve[quintile] * self.total_wealth
            qty = max(1.0, np.sum([1 for a in self.schedule.agents if a.social_stratum == quintile and a.age >= 18]))
            ag_share = total / qty
            for agent in filter(lambda x: x.social_stratum == quintile and x.age >= 18, self.schedule.agents):
                agent.wealth = ag_share

        self.running= True
        self.datacollector.collect(self)
        #self.datacollector_wealth.collect(self)

    


    def compute(self):
        R0 = 0        
        susceptible = 0
        dead = 0
        recovered = 0
        infected = 0
        severe=0

        #Calculating R0
        for agent in self.schedule.agents:
          if agent.infected_others == True:
            induced_infections= agent.induced_infections
            if induced_infections == 0:
                induced_infections = [0]
            # induced_infections_ = [value for value in induced_infections if value != 0]
            infection_array = np.array(induced_infections)
            R0 = np.average(infection_array)
            self.R0 = R0

          # Calculating Susceptible, Infected, Recoverd Agents
          if agent.state == InfectionState.RECOVERED:
            recovered += 1
          elif agent.state == InfectionState.INFECTED:
            infected += 1
            if agent.severity == InfectionSeverity.Severe:
              severe += 1
          elif agent.state == InfectionState.SUSCEPTIBLE:
            susceptible += 1

        # Updating Model params
        self.recovered = recovered
        self.infected = infected
        self.severe = severe
        self.susceptible = susceptible


        # Calculating Dead
        self.dead = len(self.dead_agents)

    def compute_wealth(self):
        """Compute Wealth of All different Economic Stratum"""

        wealth_most_poor = 0
        wealth_poor = 0
        wealth_working_class = 0
        wealth_rich = 0
        wealth_most_rich = 0

        for agent in self.schedule.agents:

            if agent.social_stratum == SocialStratum.Most_Poor :
                wealth_most_poor += agent.wealth
            elif agent.social_stratum == SocialStratum.Poor :
                wealth_poor += agent.wealth
            elif agent.social_stratum == SocialStratum.Working_class:
                wealth_working_class += agent.wealth
            elif agent.social_stratum == SocialStratum.Rich:
                wealth_rich += agent.wealth
            else :
                wealth_most_rich += agent.wealth

        self.wealth_most_poor = wealth_most_poor
        self.wealth_poor = wealth_poor
        self.wealth_working_class = wealth_working_class
        self.wealth_rich = wealth_rich
        self.wealth_most_rich = wealth_most_rich

    
   
    def get_recovery_time(self):
        return int(self.random.normalvariate(self.recovery_days,self.recovery_sd))

    def step(self):

        self.schedule.step()
        self.compute()
        self.compute_wealth()
        self.datacollector.collect(self)
        #self.datacollector_wealth.collect(self)
    
    def run_model(self, n):
        for i in range(n):
            self.step()
        self.running= False