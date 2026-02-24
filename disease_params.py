"""
Age-stratified COVID-19 disease parameters.

Sources:
  - Verity et al. (2020) Lancet Inf Dis - IFR by age
  - Ferguson et al. (2020) Imperial College Report 9
  - CDC Pandemic Planning Scenarios (2021)
  - Bi et al. (2020) - Incubation period
  - Kerr et al. (2021) Covasim - transmission dynamics

All rates are per-person probabilities unless noted otherwise.
"""

import numpy as np

# Age group boundaries: [0,10), [10,20), [20,30), ..., [70,80), [80+)
AGE_GROUPS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
              (50, 60), (60, 70), (70, 80), (80, 120)]

# Infection Fatality Rate by age group (Verity et al. 2020, updated with Levin et al. 2020)
IFR_BY_AGE = {
    (0, 10): 0.00002,
    (10, 20): 0.00006,
    (20, 30): 0.0002,
    (30, 40): 0.0005,
    (40, 50): 0.002,
    (50, 60): 0.006,
    (60, 70): 0.02,
    (70, 80): 0.06,
    (80, 120): 0.10,
}

# Hospitalization rate by age (CDC Planning Scenarios / Ferguson Report 9)
HOSPITALIZATION_RATE_BY_AGE = {
    (0, 10): 0.001,
    (10, 20): 0.003,
    (20, 30): 0.012,
    (30, 40): 0.032,
    (40, 50): 0.049,
    (50, 60): 0.102,
    (60, 70): 0.166,
    (70, 80): 0.243,
    (80, 120): 0.273,
}

# Probability of being symptomatic by age (CDC / Covasim)
SYMPTOMATIC_RATE_BY_AGE = {
    (0, 10): 0.50,
    (10, 20): 0.55,
    (20, 30): 0.60,
    (30, 40): 0.65,
    (40, 50): 0.70,
    (50, 60): 0.75,
    (60, 70): 0.80,
    (70, 80): 0.85,
    (80, 120): 0.90,
}

# Relative susceptibility by age (Covasim / Davies et al. 2020 Nature Medicine)
SUSCEPTIBILITY_BY_AGE = {
    (0, 10): 0.34,
    (10, 20): 0.67,
    (20, 30): 1.00,
    (30, 40): 1.00,
    (40, 50): 1.00,
    (50, 60): 1.00,
    (60, 70): 1.24,
    (70, 80): 1.47,
    (80, 120): 1.47,
}

# Relative transmissibility by age (children transmit less)
TRANSMISSIBILITY_BY_AGE = {
    (0, 10): 0.50,
    (10, 20): 0.75,
    (20, 30): 1.00,
    (30, 40): 1.00,
    (40, 50): 1.00,
    (50, 60): 1.00,
    (60, 70): 1.00,
    (70, 80): 1.00,
    (80, 120): 1.00,
}

# Incubation period (days): log-normal distribution (Bi et al. 2020, Li et al. 2020)
INCUBATION_PERIOD_MEAN = 5.1  # days (log-normal mean)
INCUBATION_PERIOD_SD = 1.5

# Infectious period before symptom onset (presymptomatic)
PRESYMPTOMATIC_DAYS = 2.0

# Duration of infectiousness (days) after symptom onset
INFECTIOUS_DURATION_MEAN = 10.0
INFECTIOUS_DURATION_SD = 3.0

# Time from symptom onset to hospitalization (for severe cases)
SYMPTOM_TO_HOSPITAL_MEAN = 5.0
SYMPTOM_TO_HOSPITAL_SD = 2.0

# Hospital stay duration (days)
HOSPITAL_STAY_MEAN = 8.0
HOSPITAL_STAY_SD = 3.0

# Time from hospitalization to death (for fatal cases)
HOSPITAL_TO_DEATH_MEAN = 7.0
HOSPITAL_TO_DEATH_SD = 3.0

# Immunity waning rate (days until reinfection possible)
NATURAL_IMMUNITY_DURATION_MEAN = 180  # ~6 months
NATURAL_IMMUNITY_DURATION_SD = 30

# Vaccination parameters
VACCINE_PARAMS = {
    "dose_1": {
        "efficacy_infection": 0.52,    # Reduction in infection probability
        "efficacy_symptomatic": 0.65,  # Reduction in symptomatic disease
        "efficacy_severe": 0.80,       # Reduction in severe disease
        "efficacy_death": 0.85,        # Reduction in death
        "days_to_effect": 14,          # Days after dose to reach efficacy
    },
    "dose_2": {
        "efficacy_infection": 0.79,
        "efficacy_symptomatic": 0.90,
        "efficacy_severe": 0.95,
        "efficacy_death": 0.98,
        "days_to_effect": 7,
    },
    "booster": {
        "efficacy_infection": 0.85,
        "efficacy_symptomatic": 0.95,
        "efficacy_severe": 0.98,
        "efficacy_death": 0.99,
        "days_to_effect": 7,
    },
    "waning_halflife_days": 120,  # Efficacy halves every ~4 months
}

# Age distribution for population generation (US 2020 Census-like)
AGE_DISTRIBUTION = {
    (0, 10): 0.12,
    (10, 20): 0.13,
    (20, 30): 0.14,
    (30, 40): 0.13,
    (40, 50): 0.12,
    (50, 60): 0.13,
    (60, 70): 0.12,
    (70, 80): 0.07,
    (80, 120): 0.04,
}


def get_age_group(age):
    """Return the age group tuple for a given age."""
    for group in AGE_GROUPS:
        if group[0] <= age < group[1]:
            return group
    return AGE_GROUPS[-1]  # 80+


def get_param_for_age(age, param_dict):
    """Look up an age-stratified parameter value."""
    return param_dict[get_age_group(age)]


def sample_incubation_period(rng):
    """Sample incubation period from log-normal distribution."""
    return max(1, int(rng.lognormvariate(
        np.log(INCUBATION_PERIOD_MEAN), 0.3)))


def sample_infectious_duration(rng):
    """Sample duration of infectiousness."""
    return max(3, int(rng.normalvariate(INFECTIOUS_DURATION_MEAN, INFECTIOUS_DURATION_SD)))


def sample_hospital_duration(rng):
    """Sample hospital stay duration."""
    return max(1, int(rng.normalvariate(HOSPITAL_STAY_MEAN, HOSPITAL_STAY_SD)))


def sample_age(rng):
    """Sample an age from the population distribution."""
    groups = list(AGE_DISTRIBUTION.keys())
    probs = list(AGE_DISTRIBUTION.values())
    group = rng.choices(groups, weights=probs, k=1)[0]
    return rng.uniform(group[0], group[1])


def get_vaccine_efficacy(dose_level, days_since_vaccination, param_key="efficacy_infection"):
    """
    Calculate current vaccine efficacy accounting for waning.

    Args:
        dose_level: "dose_1", "dose_2", or "booster"
        days_since_vaccination: days since this dose was administered
        param_key: which efficacy to return

    Returns:
        Current efficacy (0-1)
    """
    params = VACCINE_PARAMS[dose_level]
    if days_since_vaccination < params["days_to_effect"]:
        # Ramp up linearly during onset period
        ramp = days_since_vaccination / params["days_to_effect"]
        return params[param_key] * ramp

    # Waning: exponential decay with half-life
    days_waning = days_since_vaccination - params["days_to_effect"]
    halflife = VACCINE_PARAMS["waning_halflife_days"]
    decay = 0.5 ** (days_waning / halflife)
    return params[param_key] * decay
