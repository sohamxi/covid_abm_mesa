import enum
from enum import Enum
import numpy as np
from mesa import Agent, Model
from disease_params import (
    get_param_for_age, IFR_BY_AGE, HOSPITALIZATION_RATE_BY_AGE,
    SYMPTOMATIC_RATE_BY_AGE, SUSCEPTIBILITY_BY_AGE, TRANSMISSIBILITY_BY_AGE,
    sample_incubation_period, sample_infectious_duration, sample_age,
    get_vaccine_efficacy, NATURAL_IMMUNITY_DURATION_MEAN, NATURAL_IMMUNITY_DURATION_SD,
)
from contact_network import ContactLayer, LAYER_TRANSMISSION_MULTIPLIER, get_contacts


class InfectionState(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DIED = 3
    EXPOSED = 4


class InfectionSeverity(Enum):
    Asymptomatic = 'a'
    Quarantined = 'q'
    Severe = 's'


class JobType(Enum):
    GOVERNMENT = 'g'
    BLUE_COLLAR = 'l'
    WHITE_COLLAR = 'e'
    UNEMPLOYED = 'u'
    BUSINESS_OWNER = 'b'


class SocialStratum(enum.IntEnum):
    Most_Poor = 0
    Poor = 1
    Working_class = 2
    Rich = 3
    Most_Rich = 4


# Wealth distribution - Lorenz Curve (South American Nations)
lorenz_curve = [.04, .08, .13, .2, .55]
share = np.min(lorenz_curve)
basic_income = np.array(lorenz_curve) / share


class Human(Agent):
    """ An agent in an epidemic model."""

    def __init__(self, model):
        super().__init__(model)
        # Demographics
        self.age = sample_age(self.random)
        self.is_student = False

        # Disease state
        self.state = InfectionState.SUSCEPTIBLE
        self.severity = InfectionSeverity.Asymptomatic
        self.infection_time = 0
        self.incubation_period = 0  # Days from exposure to infectious
        self.recovery_time = 0
        self.induced_infections = 0
        self.infected_others = False
        self.symptoms = int(self.random.normalvariate(10, 4))

        # Age-stratified parameters
        self.susceptibility = get_param_for_age(self.age, SUSCEPTIBILITY_BY_AGE)
        self.transmissibility = get_param_for_age(self.age, TRANSMISSIBILITY_BY_AGE)
        self.ifr = get_param_for_age(self.age, IFR_BY_AGE)
        self.hospitalization_rate = get_param_for_age(self.age, HOSPITALIZATION_RATE_BY_AGE)
        self.symptomatic_rate = get_param_for_age(self.age, SYMPTOMATIC_RATE_BY_AGE)

        # Vaccination state
        self.vaccinated = False
        self.vaccine_dose = None  # "dose_1", "dose_2", "booster"
        self.vaccination_day = 0

        # Immunity (waning natural immunity after recovery)
        self.immunity_wane_day = 0  # Day when natural immunity expires

        # Contact network memberships (set by contact_network.build_*)
        self.household_id = -1
        self.household_members = []
        self.workplace_id = -1
        self.workplace_members = []
        self.school_id = -1
        self.school_members = []

        # Economic params
        self.social_stratum = self.random.choice([0, 1, 2, 3, 4])
        self.wealth = 0
        self.income = basic_income[self.social_stratum]
        self.expanditure = 0

    def get_vaccine_protection(self, param_key="efficacy_infection"):
        """Get current vaccine efficacy accounting for waning."""
        if not self.vaccinated or self.vaccine_dose is None:
            return 0.0
        days_since = self.model.steps - self.vaccination_day
        return get_vaccine_efficacy(self.vaccine_dose, days_since, param_key)

    def vaccinate(self, dose_level):
        """Administer a vaccine dose."""
        self.vaccinated = True
        self.vaccine_dose = dose_level
        self.vaccination_day = self.model.steps

    def getAgentIncome(self):
        step_income = 0
        basic_income_temp = 0
        variable_income_temp = 0

        if (self.state != InfectionState.DIED) and (self.severity == InfectionSeverity.Asymptomatic):
            if self.age >= 18:
                mov_prob = self.model.mov_prob
                move_today = self.random.random() < mov_prob
                if move_today:
                    basic_income_temp = basic_income[self.social_stratum]
                    variable_income_temp = self.random.random() * self.random.random() * basic_income[self.social_stratum]
        else:
            basic_income_temp = 0
            variable_income_temp = 0

        step_income = basic_income_temp + variable_income_temp
        return step_income

    def getAgentExpense(self):
        expense_temp = self.random.random() * basic_income[self.social_stratum]
        return expense_temp

    def update_Wealth(self):
        self.income = self.getAgentIncome()
        self.expanditure = self.getAgentExpense()
        self.wealth = self.wealth + self.income - self.expanditure

    def move(self):
        """Move the agent on the grid (community layer)."""
        if self.severity == InfectionSeverity.Asymptomatic:
            mov_prob = self.model.mov_prob
            move_today = self.random.random() < mov_prob
            if move_today:
                possible_steps = self.model.grid.get_neighborhood(
                    self.pos,
                    moore=True,
                    include_center=True)
                new_position = self.random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)

    def status(self):
        """Check and update infection status with age-stratified parameters."""
        if self.state == InfectionState.EXPOSED:
            # Check if incubation period has elapsed -> become infectious
            time_exposed = self.model.steps - self.infection_time
            if time_exposed >= self.incubation_period:
                self.state = InfectionState.INFECTED
                # Determine if symptomatic
                if self.random.random() < self.symptomatic_rate:
                    self.symptoms = max(1, int(self.random.normalvariate(5, 2)))
                else:
                    self.symptoms = 999  # Asymptomatic - won't trigger quarantine

        elif self.state == InfectionState.INFECTED:
            # Severe cases may die (age-stratified IFR)
            if self.severity == InfectionSeverity.Severe:
                # Daily death probability scaled from IFR over expected severe duration
                severe_duration = max(1, self.recovery_time - self.symptoms)
                daily_death_prob = min(0.99, self.ifr / max(1, severe_duration))
                # Hospital overflow triples death rate
                if self.model.severe >= self.model.hospital_capacity:
                    daily_death_prob = min(0.99, daily_death_prob * 3.0)
                # Vaccine reduces death probability
                vax_protection = self.get_vaccine_protection("efficacy_death")
                daily_death_prob *= (1 - vax_protection)
                if self.random.random() < daily_death_prob:
                    self.state = InfectionState.DIED
                    return

            # Non-severe infected may become severe (age-stratified hospitalization)
            if self.severity == InfectionSeverity.Asymptomatic or self.severity == InfectionSeverity.Quarantined:
                hosp_rate = self.hospitalization_rate
                # Vaccine reduces severe disease
                vax_protection = self.get_vaccine_protection("efficacy_severe")
                hosp_rate *= (1 - vax_protection)
                daily_severe_prob = hosp_rate / max(1, self.recovery_time)
                if self.random.random() < daily_severe_prob:
                    self.severity = InfectionSeverity.Severe

            # Symptom onset -> quarantine
            time_infected = self.model.steps - self.infection_time
            if time_infected >= self.symptoms:
                if self.severity != InfectionSeverity.Severe:
                    self.severity = InfectionSeverity.Quarantined

            # Recovery
            if time_infected >= self.recovery_time:
                self.state = InfectionState.RECOVERED
                self.severity = InfectionSeverity.Asymptomatic
                # Set immunity waning timer
                self.immunity_wane_day = self.model.steps + max(30, int(
                    self.random.normalvariate(NATURAL_IMMUNITY_DURATION_MEAN, NATURAL_IMMUNITY_DURATION_SD)))

        elif self.state == InfectionState.RECOVERED:
            self.severity = InfectionSeverity.Asymptomatic
            # Check if natural immunity has waned
            if self.model.steps >= self.immunity_wane_day:
                self.state = InfectionState.SUSCEPTIBLE

    def interact(self):
        """Multi-layer contact interaction."""
        if self.state != InfectionState.INFECTED:
            return
        if self.severity == InfectionSeverity.Severe:
            return  # Hospitalized, no community contacts

        mov_prob = self.model.mov_prob

        # Household contacts (always active, even under quarantine/lockdown)
        hh_contacts = get_contacts(self, ContactLayer.HOUSEHOLD, self.random)
        for other in hh_contacts:
            if other.state != InfectionState.DIED:
                self.infect(other, ContactLayer.HOUSEHOLD)

        # If quarantined, skip workplace/school/community
        if self.severity == InfectionSeverity.Quarantined:
            return

        # Workplace contacts (gated by mobility — lockdown keeps people home)
        if self.random.random() < mov_prob:
            wp_contacts = get_contacts(self, ContactLayer.WORKPLACE, self.random)
            for other in wp_contacts:
                if other.state != InfectionState.DIED:
                    self.infect(other, ContactLayer.WORKPLACE)

        # School contacts (gated by mobility — schools close during lockdown)
        if self.random.random() < mov_prob:
            sc_contacts = get_contacts(self, ContactLayer.SCHOOL, self.random)
            for other in sc_contacts:
                if other.state != InfectionState.DIED:
                    self.infect(other, ContactLayer.SCHOOL)

        # Community contacts (grid-based, gated by mobility)
        if self.random.random() < mov_prob and self.pos is not None:
            neighbors = self.model.grid.get_cell_list_contents([self.pos])
            for other in neighbors:
                if other is not self and other.state != InfectionState.DIED:
                    self.infect(other, ContactLayer.COMMUNITY)

    def infect(self, other, layer=ContactLayer.COMMUNITY):
        """Attempt to infect another agent, with layer-specific and age-stratified probabilities."""
        if self.state != InfectionState.INFECTED:
            return

        # Base transmission probability * layer modifier * sender transmissibility
        ptrans = self.model.ptrans
        ptrans *= LAYER_TRANSMISSION_MULTIPLIER[layer]
        ptrans *= self.transmissibility

        # Receiver susceptibility
        ptrans *= other.susceptibility

        # Vaccine protection for receiver
        vax_protection = other.get_vaccine_protection("efficacy_infection")
        ptrans *= (1 - vax_protection)

        if other.state == InfectionState.SUSCEPTIBLE or other.state == InfectionState.EXPOSED:
            if self.random.random() < ptrans:
                if other.state == InfectionState.SUSCEPTIBLE:
                    # New exposure — severity develops later during infection
                    other.state = InfectionState.EXPOSED
                    other.infection_time = self.model.steps
                    other.incubation_period = sample_incubation_period(self.random)
                    other.recovery_time = other.incubation_period + sample_infectious_duration(self.random)
                    self.induced_infections += 1
                    self.infected_others = True
                elif other.state == InfectionState.EXPOSED:
                    # Already exposed, just count secondary infection attempt
                    pass
            # If ptrans roll fails, susceptible stays susceptible (no automatic EXPOSED)

        elif other.state == InfectionState.RECOVERED:
            # Reinfection (modulated by waning immunity)
            reinfection_ptrans = ptrans * self.model.reinfection_rate
            if self.random.random() < reinfection_ptrans:
                other.state = InfectionState.EXPOSED
                other.infection_time = self.model.steps
                other.incubation_period = sample_incubation_period(self.random)
                other.recovery_time = other.incubation_period + sample_infectious_duration(self.random)
                self.induced_infections += 1
                self.infected_others = True

    def step(self):
        if self.state == InfectionState.DIED:
            return
        self.status()
        self.move()
        self.interact()
        self.update_Wealth()
