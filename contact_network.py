"""
Multi-layer contact network for COVID-19 ABM.

Layers:
  - Household: Small groups (2-6 people), high contact rate, always active
  - Workplace: Medium groups (5-30 people), active for working-age adults
  - School: Medium groups (15-35 people), active for children/young adults
  - Community: Random grid-based encounters, lower transmission probability

Each layer has its own transmission probability modifier, reflecting
different contact durations and proximity levels.

References:
  - Ferguson et al. (2020) Imperial College Report 9
  - Covasim: Kerr et al. (2021) PLOS Computational Biology
  - Mossong et al. (2008) POLYMOD contact survey
"""

import numpy as np
from enum import Enum


class ContactLayer(Enum):
    HOUSEHOLD = "household"
    WORKPLACE = "workplace"
    SCHOOL = "school"
    COMMUNITY = "community"


# Relative transmission probability per layer (multiply by base ptrans)
# Household contacts are long-duration, close proximity -> highest
# Community contacts are brief, random -> lowest
LAYER_TRANSMISSION_MULTIPLIER = {
    ContactLayer.HOUSEHOLD: 3.0,   # Long-duration close contact (SAR ~18%)
    ContactLayer.WORKPLACE: 0.6,   # Covasim calibrated value
    ContactLayer.SCHOOL: 0.6,      # Covasim calibrated value
    ContactLayer.COMMUNITY: 0.3,   # Brief random encounters
}

# Average number of daily contacts per layer (from POLYMOD / Covasim)
LAYER_DAILY_CONTACTS = {
    ContactLayer.HOUSEHOLD: None,  # All household members (determined by group size)
    ContactLayer.WORKPLACE: 8,
    ContactLayer.SCHOOL: 12,
    ContactLayer.COMMUNITY: 4,
}


def build_households(agents, rng):
    """
    Assign agents to households.

    Household size distribution based on census data (US/European average):
    1-person: 28%, 2-person: 34%, 3-person: 16%, 4-person: 14%, 5+: 8%
    """
    household_sizes = []
    size_probs = [0.28, 0.34, 0.16, 0.14, 0.08]
    size_values = [1, 2, 3, 4, 5]

    shuffled = list(agents)
    rng.shuffle(shuffled)

    households = []
    idx = 0
    while idx < len(shuffled):
        size = rng.choices(size_values, weights=size_probs, k=1)[0]
        size = min(size, len(shuffled) - idx)
        household = shuffled[idx:idx + size]
        households.append(household)
        for agent in household:
            agent.household_id = len(households) - 1
            agent.household_members = household
        idx += size

    return households


def build_workplaces(agents, rng):
    """
    Assign working-age adults (18-65) to workplaces.
    Workplace size: 5-30 people.
    """
    workers = [a for a in agents if 18 <= a.age <= 65 and not getattr(a, 'is_student', False)]
    rng.shuffle(workers)

    workplaces = []
    idx = 0
    while idx < len(workers):
        size = rng.randint(5, 30)
        size = min(size, len(workers) - idx)
        workplace = workers[idx:idx + size]
        workplaces.append(workplace)
        for agent in workplace:
            agent.workplace_id = len(workplaces) - 1
            agent.workplace_members = workplace
        idx += size

    return workplaces


def build_schools(agents, rng):
    """
    Assign school-age agents (5-22) to school classes.
    Class size: 15-35 students.
    """
    students = [a for a in agents if 5 <= a.age <= 22]
    rng.shuffle(students)

    for s in students:
        s.is_student = True

    schools = []
    idx = 0
    while idx < len(students):
        size = rng.randint(15, 35)
        size = min(size, len(students) - idx)
        school_class = students[idx:idx + size]
        schools.append(school_class)
        for agent in school_class:
            agent.school_id = len(schools) - 1
            agent.school_members = school_class
        idx += size

    return schools


def get_contacts(agent, layer, rng, max_contacts=None):
    """
    Get contacts for an agent in a given layer.

    Returns a list of agents this agent will interact with during this step.
    """
    if layer == ContactLayer.HOUSEHOLD:
        # Contact all household members
        return [a for a in getattr(agent, 'household_members', []) if a is not agent]

    elif layer == ContactLayer.WORKPLACE:
        members = getattr(agent, 'workplace_members', [])
        if not members:
            return []
        others = [a for a in members if a is not agent]
        n_contacts = min(len(others), max_contacts or LAYER_DAILY_CONTACTS[ContactLayer.WORKPLACE])
        if n_contacts <= 0:
            return []
        return rng.sample(others, min(n_contacts, len(others)))

    elif layer == ContactLayer.SCHOOL:
        members = getattr(agent, 'school_members', [])
        if not members:
            return []
        others = [a for a in members if a is not agent]
        n_contacts = min(len(others), max_contacts or LAYER_DAILY_CONTACTS[ContactLayer.SCHOOL])
        if n_contacts <= 0:
            return []
        return rng.sample(others, min(n_contacts, len(others)))

    elif layer == ContactLayer.COMMUNITY:
        # Grid-based: return current cell neighbors (already handled by grid)
        return None  # Signal to use grid-based contacts

    return []
