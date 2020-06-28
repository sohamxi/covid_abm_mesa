import time,enum
from enum import Enum
import numpy as np
import pandas as pd
import pylab as plt
from mesa import Agent, Model

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
    Quarantined= 'q'
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
        self.severity = InfectionSeverity.Asymptomatic
        # TODO - Job type to affect Income and thus wealth
        #self.jobtype = JobType.WHITE_COLLAR
        #self.jobtype = np.random.choice([JobType.GOVERNMENT,JobType.BLUE_COLLAR,JobType.WHITE_COLLARJobType.UNEMPLOYED,JobType.BUSINESS_OWNER]) #p=[0.2,0.2,0.2,0.2,0.2] (Optional Based on Demography)
        self.infection_time = 0
        self.induced_infections = 0
        self.infected_others = False
        self.symptoms = int(self.random.normalvariate(10,4))
        # Economic params
        self.social_stratum = np.random.choice([0,1,2,3,4])
        self.wealth = 0
        self.income = basic_income[self.social_stratum]
        self.expanditure = 0

    def getAgentIncome(self):
        """Calculate Agent's Income for the Step"""

        step_income =0
        basic_income_temp = 0
        variable_income_temp = 0

        if (self.state != InfectionState.DIED) and (self.severity == InfectionSeverity.Asymptomatic) :
            
            if self.age >= 18:
                mov_prob = self.model.mov_prob
                move_today = np.random.choice([True,False],p=[mov_prob,1-mov_prob])
                if move_today:
                    basic_income_temp = basic_income[self.social_stratum]
                    variable_income_temp = self.random.random() *self.random.random()* basic_income[self.social_stratum]
        else:
            basic_income_temp = 0
            variable_income_temp = 0

        step_income = basic_income_temp + variable_income_temp
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
        if self.severity == InfectionSeverity.Asymptomatic:
            mov_prob = self.model.mov_prob
            move_today = np.random.choice([True,False],p=[mov_prob,1-mov_prob])
            if move_today:
                possible_steps = self.model.grid.get_neighborhood(
                    self.pos,
                    moore=True,
                    include_center=True)
                new_position = self.random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)

    
    
    
    def status(self):
        """Check infection status"""

        if self.state == InfectionState.INFECTED:

            

            ## Some of the severe people die     
            if self.severity == InfectionSeverity.Severe:
                cond_drate = self.model.death_rate/ self.model.severe_perc
                if self.recovery_time <= 1:
                    self.recovery_time =1
                drate = cond_drate/self.recovery_time
                #if drate >= 1:
                #print(f'death rate is too high : {drate} Agent:{self.unique_id}')
                #drate = self.model.death_rate/ self.model.severe_perc
                #print(f'For Agent : {self.unique_id} Death Rate:{drate}')
                alive = np.random.choice([0,1], p=[drate,1-drate])
                # Check wheather Alive, if Dead remove from scheduler
                if alive == 0:
                    self.state = InfectionState.DIED
                    self.model.schedule.remove(self)
                    self.model.dead_agents.append(self.unique_id)
            
            ## Some of Infected but Asymptomatic people become Severe
            if self.severity == InfectionSeverity.Asymptomatic or self.severity ==InfectionSeverity.Quarantined:
                ## TODO: Change probability for Asymptomatic & Quarantine
                turn_severe_prob = self.model.severe_perc/max(1, self.recovery_time)
                #print(f'Might turn severe with probability {turn_severe_prob}')
                turn_severe_today = np.random.choice([True,False],p=[turn_severe_prob,1-turn_severe_prob])
                if turn_severe_today:
                    self.severity = InfectionSeverity.Severe
                    #print(f'Agent {self.unique_id} has turned Severe with Proba: {turn_severe_prob}')

            #  People Passed due time show symptoms and Put to Quarantine
            time_passed = self.model.schedule.time - self.infection_time
            if time_passed >= self.symptoms:
                self.severity = InfectionSeverity.Quarantined
                #print(f'Agent {self.unique_id} has been put to quarentine')
            
            #People passed recovery date recovered 
            if time_passed >= self.recovery_time:
                #print(f'Agent {self.unique_id} recovered with Time Passed:{time_passed} & Assigned time for recovery:{self.recovery_time}')          
                self.state = InfectionState.RECOVERED
                self.severity = InfectionSeverity.Asymptomatic
        elif self.state == InfectionState.RECOVERED:
            self.severity = InfectionSeverity.Asymptomatic
    
    
    
    def interact(self):
        """Interaction with other Agents"""

        if self.severity == InfectionSeverity.Asymptomatic:
            neighbors = self.model.grid.get_cell_list_contents([self.pos])

            if len(neighbors)==0:
                print (self.unique_id + " is lonely")
            else :
                for other in neighbors:
                    if other.state == InfectionState.DIED:
                        pass
                    else:
                        self.infect(other)

        elif self.severity == InfectionSeverity.Quarantined:
            #print(f'Agent {self.unique_id} is Quarantined with Severity {self.severity}')
            pass
        else:
            #print(f'Agent {self.unique_id} is Hospitalised with Severity {self.severity}')
            pass




    def infect(self,other):
        """Infect/Reinfect Other Agent"""

        if self.state == InfectionState.INFECTED :
          # check other Agent is Susceptible or Recovered : Susceptible
          if other.state == InfectionState.SUSCEPTIBLE or other.state == InfectionState.EXPOSED:
            if self.random.random() < self.model.ptrans:
              other.state = InfectionState.INFECTED
              other.infection_time = self.model.schedule.time
              other.recovery_time = self.model.get_recovery_time()
              #print(f'New Person Infected, recovery rate : {other.recovery_time} and time till symptom : {other.symptoms}') 
              self.induced_infections +=1
              self.infected_others = True
              # set Severity
              if self.random.random() < self.model.severe_perc:
                other.severity = InfectionSeverity.Severe
            else:
                other.state = InfectionState.EXPOSED
            #   else :
            #     other.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])


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
            #   else :
            #     other.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])


    def step(self):
        self.status()
        self.move()
        self.interact()
        self.update_Wealth()