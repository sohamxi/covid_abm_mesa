import time,enum
from enum import Enum
import numpy as np
import pandas as pd
import pylab as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import *


class InfectionModel(Model):

    def __init__(self, N=10, width=10, height=10, ptrans = 0.25, reinfection_rate = 0.00,  severe_perc =0.18,
                  death_rate = 0.0193, recovery_days = 21,
                 recovery_sd = 7, initial_infected_perc=0.1,  
                 lockdown = False, saq = False, ipa = False, mm= False, hospital_capacity = 0.01
                 ):
        self.population = N
        #self.model = model
        self.ptrans = ptrans
        self.reinfection_rate = reinfection_rate
        self.mov_prob = 1
        self.death_rate = death_rate
        self.initial_death_rate = death_rate
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.dead_agents = []

        self.severe_perc = severe_perc
        self.initial_infected_perc = initial_infected_perc        
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        
        self.hospital_capacity = hospital_capacity*self.population
        self.susceptible = self.population
        self.dead = 0
        self.recovered = 0
        self.infected = 0
        self.R0 = 0
        self.severe = 0
        self.exposed = 0
        # Calculating percentages
        self.percentage_susceptible = (self.susceptible/self.population)*100
        self.percentage_dead = (self.dead/self.population)*100
        self.percentage_recovered = (self.recovered/self.population)*100
        self.percentage_infected = (self.infected/self.population)*100
        self.percentage_exposed = (self.exposed/self.population)*100
        self.percentage_severe = (self.severe/self.population)*100
        # Economic params Model related
        self.total_wealth = 10**4
        self.wealth_most_poor = lorenz_curve[0] * self.total_wealth
        self.wealth_poor = lorenz_curve[1] * self.total_wealth
        self.wealth_working_class = lorenz_curve[2] * self.total_wealth
        self.wealth_rich = lorenz_curve[3] * self.total_wealth
        self.wealth_most_rich = lorenz_curve[4] * self.total_wealth

        # Making Provision for Interventions
        self.intervention1 = lockdown
        self.intervention2 = saq
        self.intervention3 = ipa
        self.intervention4 = mm
        print(f'Intervention Sattus: \n Lockdown:{self.intervention1}; Screening:{self.intervention2}, Public Awareness:{self.intervention3}; Masks:{self.intervention4}')

        # Create Data Collecter for Aggregate Values  
        self.datacollector = DataCollector(model_reporters={"infected": 'percentage_infected',
                                                            "recovered": 'percentage_recovered',
                                                            "susceptible": 'percentage_susceptible',
                                                            "exposed": 'percentage_exposed',
                                                            "dead": 'dead',
                                                            "R0": 'R0',
                                                            "hospital" : "hospital_capacity",
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
                # else :
                #   a.severity = np.random.choice([InfectionSeverity.Asymptomatic,InfectionSeverity.Hospitalization])
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
        exposed = 0
        dead = 0
        recovered = 0
        infected = 0
        severe=0
        infection_array =[]

        #Calculating R0
        for agent in self.schedule.agents:
          if agent.infected_others == True:
            induced_infections= agent.induced_infections
            if induced_infections == 0:
                induced_infections = [0]
            # induced_infections_ = [value for value in induced_infections if value != 0]
            infection_array.append(induced_infections)
            
            

          # Calculating Susceptible, Infected, Recoverd Agents
          if agent.state == InfectionState.RECOVERED:
            recovered += 1
          elif agent.state == InfectionState.INFECTED:
            infected += 1
            if agent.severity == InfectionSeverity.Severe:
              severe += 1
          elif agent.state == InfectionState.SUSCEPTIBLE:
            susceptible += 1
          elif agent.state == InfectionState.EXPOSED :
            exposed += 1


        # Updating Model params
        #print(infection_array)
        R0 = np.average(infection_array)
        self.R0 = R0
        self.recovered = recovered
        self.infected = infected
        self.severe = severe
        self.susceptible = susceptible
        self.exposed = exposed
        if self.severe >= self.hospital_capacity:
            # If Severity exceeds Healthcare Capaity Death Rate will increase
            self.death_rate = self.initial_death_rate * 3
            #print(f'Death rate updated to :{self.death_rate}')
        else:
            self.death_rate = self.initial_death_rate
            #print(f'Death rate updated to :{self.death_rate}')
        

        # Calculating Dead
        self.dead = len(self.dead_agents)

        # Calculating percentages
        self.percentage_susceptible = (self.susceptible/self.population)*100
        self.percentage_dead = (self.dead/self.population)*100
        self.percentage_recovered = (self.recovered/self.population)*100
        self.percentage_infected = (self.infected/self.population)*100
        self.percentage_exposed = (self.exposed/self.population)*100
        self.percentage_severe = (self.severe/self.population)*100

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


    def apply_lockdown(self):
        if self.infected >= self.population *0.1: 
            self.mov_prob = 0.1
            print(f'#of Infecetd: {self.infected} LockDown Imposed!!!')

    def apply_quarantine(self):
        """Show Symptoms for all infected agents immediately"""
        ### TODO: Should it apply on each steps

        for agent in self.schedule.agents:
            if agent.state == InfectionState.INFECTED:
                agent.symptoms = 6*0.5 # min(symptoms)*0.5
                #print(f'Agent with id: {agent.unique_id} will be shown symptoms in:{agent.symptoms}')
        #self.agent.symptoms=0

    def check_for_intervention(self):
        if self.intervention1:
            self.apply_lockdown()
        if self.intervention2:
            self.apply_quarantine()
        if self.intervention3:
            self.ptrans *= 0.3 
        if self.intervention4:
            self.ptrans *= 0.2

    def step(self):
        self.check_for_intervention()
        self.schedule.step()
        self.compute()
        self.compute_wealth()
        self.datacollector.collect(self)
        #self.datacollector_wealth.collect(self)
        if self.schedule.time == 60:
            self.running = False

    # def run_model(self, n):
    #     for i in range(n):
    #         self.step()
    #     self.running= False