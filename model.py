import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import *
from contact_network import build_households, build_workplaces, build_schools


class InfectionModel(Model):

    def __init__(self, N=10, width=10, height=10, ptrans=0.05, reinfection_rate=0.02, severe_perc=0.18,
                 death_rate=0.0193, recovery_days=21,
                 recovery_sd=7, initial_infected_perc=0.1,
                 lockdown=False, saq=False, ipa=False, mm=False,
                 vaccination=False, vaccination_rate=0.01,
                 hospital_capacity=0.01,
                 seed=None):
        super().__init__(seed=seed)
        self.population = N
        self.ptrans = ptrans
        self.initial_ptrans = ptrans
        self.reinfection_rate = reinfection_rate
        self.mov_prob = 1
        self.death_rate = death_rate
        self.initial_death_rate = death_rate
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd

        self.severe_perc = severe_perc
        self.initial_infected_perc = initial_infected_perc
        self.grid = MultiGrid(width, height, True)

        self.hospital_capacity = hospital_capacity * self.population
        self.susceptible = self.population
        self.dead = 0
        self.recovered = 0
        self.infected = 0
        self.R0 = 0
        self.severe = 0
        self.exposed = 0
        self.vaccinated_count = 0
        # Calculating percentages
        self.percentage_susceptible = (self.susceptible / self.population) * 100
        self.percentage_dead = (self.dead / self.population) * 100
        self.percentage_recovered = (self.recovered / self.population) * 100
        self.percentage_infected = (self.infected / self.population) * 100
        self.percentage_exposed = (self.exposed / self.population) * 100
        self.percentage_severe = (self.severe / self.population) * 100
        self.percentage_vaccinated = 0
        # Economic params
        self.total_wealth = 10 ** 4
        self.wealth_most_poor = lorenz_curve[0] * self.total_wealth
        self.wealth_poor = lorenz_curve[1] * self.total_wealth
        self.wealth_working_class = lorenz_curve[2] * self.total_wealth
        self.wealth_rich = lorenz_curve[3] * self.total_wealth
        self.wealth_most_rich = lorenz_curve[4] * self.total_wealth

        # Interventions
        self.intervention1 = lockdown
        self.intervention2 = saq
        self.intervention3 = ipa
        self.intervention4 = mm
        self.vaccination_enabled = vaccination
        self.vaccination_rate = vaccination_rate  # Fraction of susceptible vaccinated per step
        print(f'Intervention Status: \n Lockdown:{self.intervention1}; Screening:{self.intervention2}, '
              f'Public Awareness:{self.intervention3}; Masks:{self.intervention4}; Vaccination:{self.vaccination_enabled}')

        # Data Collector
        self.datacollector = DataCollector(model_reporters={
            "infected": 'percentage_infected',
            "recovered": 'percentage_recovered',
            "susceptible": 'percentage_susceptible',
            "exposed": 'percentage_exposed',
            "dead": 'dead',
            "R0": 'R0',
            "hospital": "hospital_capacity",
            "severe_cases": 'severe',
            "vaccinated": 'percentage_vaccinated',
            "Most Poor": 'wealth_most_poor',
            "Poor": 'wealth_poor',
            "Middle Class": 'wealth_working_class',
            "Rich": 'wealth_rich',
            "Most Rich": 'wealth_most_rich'
        })

        # Create Agents
        for i in range(self.population):
            a = Human(self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            # Initial Infection
            infection = np.random.choice([0, 1], p=[(1 - self.initial_infected_perc), self.initial_infected_perc])
            if infection == 1:
                a.state = InfectionState.INFECTED
                a.infection_time = 0
                a.recovery_time = self.get_recovery_time()
                severity = np.random.choice([0, 1], p=[(1 - self.severe_perc), self.severe_perc])
                if severity == 1:
                    a.severity = InfectionSeverity.Severe

        # Build contact networks
        agent_list = list(self.agents)
        self.households = build_households(agent_list, self.random)
        self.schools = build_schools(agent_list, self.random)
        self.workplaces = build_workplaces(agent_list, self.random)

        # Wealth Distribution
        for quintile in [0, 1, 2, 3, 4]:
            total = lorenz_curve[quintile] * self.total_wealth
            qty = max(1.0, np.sum([1 for a in self.agents if a.social_stratum == quintile and a.age >= 18]))
            ag_share = total / qty
            for agent in filter(lambda x: x.social_stratum == quintile and x.age >= 18, self.agents):
                agent.wealth = ag_share

        self.running = True
        self.datacollector.collect(self)

    def compute(self):
        susceptible = 0
        exposed = 0
        dead = 0
        recovered = 0
        infected = 0
        severe = 0
        vaccinated_count = 0
        infection_array = []

        for agent in self.agents:
            if agent.infected_others:
                infection_array.append(agent.induced_infections)

            if agent.state == InfectionState.RECOVERED:
                recovered += 1
            elif agent.state == InfectionState.INFECTED:
                infected += 1
                if agent.severity == InfectionSeverity.Severe:
                    severe += 1
            elif agent.state == InfectionState.SUSCEPTIBLE:
                susceptible += 1
            elif agent.state == InfectionState.EXPOSED:
                exposed += 1
            elif agent.state == InfectionState.DIED:
                dead += 1

            if agent.vaccinated:
                vaccinated_count += 1

        self.R0 = np.average(infection_array) if infection_array else 0
        self.recovered = recovered
        self.infected = infected
        self.severe = severe
        self.susceptible = susceptible
        self.exposed = exposed
        self.dead = dead
        self.vaccinated_count = vaccinated_count

        if self.severe >= self.hospital_capacity:
            self.death_rate = self.initial_death_rate * 3
        else:
            self.death_rate = self.initial_death_rate

        self.percentage_susceptible = (self.susceptible / self.population) * 100
        self.percentage_dead = (self.dead / self.population) * 100
        self.percentage_recovered = (self.recovered / self.population) * 100
        self.percentage_infected = (self.infected / self.population) * 100
        self.percentage_exposed = (self.exposed / self.population) * 100
        self.percentage_severe = (self.severe / self.population) * 100
        self.percentage_vaccinated = (self.vaccinated_count / self.population) * 100

    def compute_wealth(self):
        wealth_most_poor = 0
        wealth_poor = 0
        wealth_working_class = 0
        wealth_rich = 0
        wealth_most_rich = 0

        for agent in self.agents:
            if agent.social_stratum == SocialStratum.Most_Poor:
                wealth_most_poor += agent.wealth
            elif agent.social_stratum == SocialStratum.Poor:
                wealth_poor += agent.wealth
            elif agent.social_stratum == SocialStratum.Working_class:
                wealth_working_class += agent.wealth
            elif agent.social_stratum == SocialStratum.Rich:
                wealth_rich += agent.wealth
            else:
                wealth_most_rich += agent.wealth

        self.wealth_most_poor = wealth_most_poor
        self.wealth_poor = wealth_poor
        self.wealth_working_class = wealth_working_class
        self.wealth_rich = wealth_rich
        self.wealth_most_rich = wealth_most_rich

    def get_recovery_time(self):
        return int(self.random.normalvariate(self.recovery_days, self.recovery_sd))

    def apply_lockdown(self):
        """Graduated lockdown: mobility drops proportional to infection level."""
        infection_pct = self.infected / max(1, self.population)
        if infection_pct >= 0.1:
            self.mov_prob = 0.1
        elif infection_pct >= 0.05:
            self.mov_prob = 0.3
        elif infection_pct >= 0.02:
            self.mov_prob = 0.5
        else:
            self.mov_prob = 1.0

    def apply_quarantine(self):
        """Screening: reduce time to symptom detection for all infected."""
        for agent in self.agents:
            if agent.state == InfectionState.INFECTED:
                agent.symptoms = min(agent.symptoms, 3)

    def apply_vaccination(self):
        """Vaccinate a fraction of susceptible agents each step."""
        if not self.vaccination_enabled:
            return

        # Prioritize elderly, then general population
        susceptible_agents = [a for a in self.agents
                              if a.state == InfectionState.SUSCEPTIBLE
                              and not a.vaccinated]

        # Sort by age descending (elderly first)
        susceptible_agents.sort(key=lambda a: a.age, reverse=True)

        n_to_vaccinate = max(1, int(self.vaccination_rate * len(susceptible_agents)))
        for agent in susceptible_agents[:n_to_vaccinate]:
            agent.vaccinate("dose_1")

        # Upgrade existing dose_1 to dose_2 after 21 days
        for agent in self.agents:
            if (agent.vaccinated and agent.vaccine_dose == "dose_1"
                    and self.steps - agent.vaccination_day >= 21):
                agent.vaccinate("dose_2")

    def check_for_intervention(self):
        if self.intervention1:
            self.apply_lockdown()
        else:
            self.mov_prob = 1.0

        if self.intervention2:
            self.apply_quarantine()

        if self.intervention3:
            # Public awareness: reduce transmission by 30% (applied once, not compounding)
            self.ptrans = self.initial_ptrans * 0.7
        elif not self.intervention4:
            self.ptrans = self.initial_ptrans

        if self.intervention4:
            # Masks: reduce transmission by 50%
            if self.intervention3:
                self.ptrans = self.initial_ptrans * 0.7 * 0.5
            else:
                self.ptrans = self.initial_ptrans * 0.5

        self.apply_vaccination()

    def step(self):
        self.check_for_intervention()
        self.agents.shuffle_do("step")
        self.compute()
        self.compute_wealth()
        self.datacollector.collect(self)
        if self.steps >= 60:
            self.running = False
